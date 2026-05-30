import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any

from src.training.losses import AnatomicalLoss, UncertaintyWeighting
from src.models.layers import SoftArgmax2D
from src.training.standard_trainer import generate_pytorch_heatmaps


class PoseLightningModule(pl.LightningModule):
    """
    Standard PyTorch Lightning Module wrapping the pose estimation architectures (ViTPose, HRNet)
    and consolidating all the training steps, loss terms, and optimizer construction logic.
    """

    def __init__(
        self, model: nn.Module, config: Dict[str, Any], criterion: nn.Module = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()

        training_cfg = config.get("training", {})
        self.unfreeze_epoch = training_cfg.get("unfreeze_epoch", None)
        self.backbone_lr_ratio = training_cfg.get("backbone_lr_ratio", 1.0)
        self.lambda_anatomical = training_cfg.get("lambda_anatomical", 0.0)
        self.warmup_epochs = training_cfg.get("warmup_epochs", 10)
        self.anatomical_mode = training_cfg.get("anatomical_mode", "hinge")
        self.lambda_coord = training_cfg.get("lambda_coord", 0.0)
        self.lambda_coord_occluded = training_cfg.get("lambda_coord_occluded", 0.0)
        self.sigma_start = training_cfg.get("sigma_start", 2.0)
        self.sigma_end = training_cfg.get("sigma_end", 2.0)

        # Flag indicating whether we are currently using model's own coordinate outputs
        self.using_model_coords = False

        if (
            self.lambda_anatomical > 0
            or self.lambda_coord > 0
            or self.lambda_coord_occluded > 0
        ):
            # Use high temperature to ensure soft-argmax focuses on the actual peak
            self.soft_argmax = SoftArgmax2D(temperature=100.0)

        if self.lambda_anatomical > 0:
            self.anatomical_criterion = AnatomicalLoss(
                device="cpu", mode=self.anatomical_mode
            )

        # Multi-task uncertainty weighting
        self.use_uncertainty = training_cfg.get("use_uncertainty_weighting", False)
        if self.use_uncertainty:
            # Determine tasks
            self.tasks = ["pose"]
            if self.lambda_coord > 0:
                self.tasks.append("coord_vis")
            if self.lambda_coord_occluded > 0:
                self.tasks.append("coord_occ")
            if self.lambda_anatomical > 0:
                self.tasks.append("ana")

            self.uncertainty_loss = UncertaintyWeighting(len(self.tasks))

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _get_current_lambda_ana(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.lambda_anatomical
        if epoch < self.warmup_epochs:
            return self.lambda_anatomical * (epoch / self.warmup_epochs)
        return self.lambda_anatomical

    def _get_current_sigma(self, epoch: int) -> float:
        num_epochs = self.config.get("training", {}).get("epochs", 30)
        if num_epochs <= 1:
            return self.sigma_start
        # Linear decay
        progress = min(epoch / (num_epochs * 0.7), 1.0)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        images = batch["image"]
        joints = batch["joints"]  # (B, 3, 14)

        # Dynamic Sigma Curriculum
        sigma = self._get_current_sigma(self.current_epoch)

        # Dynamically set dataset sigma for matching the curriculum
        # PL loader accesses the dataset directly.
        if (
            hasattr(self.trainer, "train_dataloader")
            and self.trainer.train_dataloader is not None
        ):
            dataset = getattr(self.trainer.train_dataloader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_sigma"):
                dataset.set_sigma(sigma)

        # Generate target heatmaps dynamically
        targets = generate_pytorch_heatmaps(
            joints=joints, heatmap_size=(64, 64), image_size=(256, 256), sigma=sigma
        )

        # Forward pass
        model_to_call = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if (
            hasattr(model_to_call, "forward")
            and "return_refined" in model_to_call.forward.__code__.co_varnames
        ):
            outputs, pred_coords = self.model(images, return_refined=True)
            self.using_model_coords = True
        else:
            outputs = self.model(images)
            self.using_model_coords = False

        loss_pose = self.criterion(outputs, targets)
        metrics = {"loss_pose": loss_pose.item(), "sigma": sigma}

        if self.use_uncertainty:
            raw_losses = {"pose": loss_pose}
        else:
            loss = loss_pose

        # 1. Coordinate regression loss
        if self.lambda_coord > 0 or self.lambda_coord_occluded > 0:
            if not self.using_model_coords:
                pred_coords = self.soft_argmax(outputs)

            gt_coords = joints[:, :2, :].permute(0, 2, 1)
            visibility = joints[:, 2, :]

            mask_vis = (visibility == 0).unsqueeze(-1).float()
            mask_occ = (visibility == 1).unsqueeze(-1).float()
            l1_all = F.l1_loss(pred_coords, gt_coords, reduction="none")

            loss_coord_vis = torch.sum(l1_all * mask_vis) / (torch.sum(mask_vis) + 1e-6)
            loss_coord_occ = torch.sum(l1_all * mask_occ) / (torch.sum(mask_occ) + 1e-6)

            metrics["loss_coord_vis"] = loss_coord_vis.item()
            metrics["loss_coord_occ"] = loss_coord_occ.item()

            if not self.use_uncertainty:
                loss = (
                    loss
                    + self.lambda_coord * loss_coord_vis
                    + self.lambda_coord_occluded * loss_coord_occ
                )
            else:
                if self.lambda_coord > 0:
                    raw_losses["coord_vis"] = loss_coord_vis
                if self.lambda_coord_occluded > 0:
                    raw_losses["coord_occ"] = loss_coord_occ

        # 2. Anatomical consistency loss
        if self.lambda_anatomical > 0:
            curr_lambda = self._get_current_lambda_ana(self.current_epoch)
            if not self.using_model_coords:
                pred_coords = self.soft_argmax(outputs)
            loss_ana = self.anatomical_criterion(pred_coords)
            metrics["loss_ana"] = loss_ana.item()
            metrics["lambda_ana"] = curr_lambda

            if not self.use_uncertainty:
                loss = loss + curr_lambda * loss_ana
            else:
                raw_losses["ana"] = loss_ana

        # 3. Final weighted loss
        if self.use_uncertainty:
            loss, weighted_metrics = self.uncertainty_loss(raw_losses)
            metrics.update(weighted_metrics)

        metrics["loss"] = loss.item()

        # Log to lightning
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        # Stash batch level metrics for Telemetry Callback
        self.last_step_metrics = metrics
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        images = batch["image"]
        joints = batch["joints"]

        sigma = self._get_current_sigma(self.current_epoch)

        targets = generate_pytorch_heatmaps(
            joints=joints, heatmap_size=(64, 64), image_size=(256, 256), sigma=sigma
        )

        model_to_call = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if (
            hasattr(model_to_call, "forward")
            and "return_refined" in model_to_call.forward.__code__.co_varnames
        ):
            outputs, pred_coords = self.model(images, return_refined=True)
            self.using_model_coords = True
        else:
            outputs = self.model(images)
            self.using_model_coords = False

        loss_pose = self.criterion(outputs, targets)
        metrics = {"loss_pose": loss_pose.item(), "sigma": sigma}

        if self.use_uncertainty:
            raw_losses = {"pose": loss_pose}
        else:
            loss = loss_pose

        if self.lambda_coord > 0 or self.lambda_coord_occluded > 0:
            if not self.using_model_coords:
                pred_coords = self.soft_argmax(outputs)

            gt_coords = joints[:, :2, :].permute(0, 2, 1)
            visibility = joints[:, 2, :]

            mask_vis = (visibility == 0).unsqueeze(-1).float()
            mask_occ = (visibility == 1).unsqueeze(-1).float()
            l1_all = F.l1_loss(pred_coords, gt_coords, reduction="none")

            loss_coord_vis = torch.sum(l1_all * mask_vis) / (torch.sum(mask_vis) + 1e-6)
            loss_coord_occ = torch.sum(l1_all * mask_occ) / (torch.sum(mask_occ) + 1e-6)

            metrics["loss_coord_vis"] = loss_coord_vis.item()
            metrics["loss_coord_occ"] = loss_coord_occ.item()

            if not self.use_uncertainty:
                loss = (
                    loss
                    + self.lambda_coord * loss_coord_vis
                    + self.lambda_coord_occluded * loss_coord_occ
                )
            else:
                if self.lambda_coord > 0:
                    raw_losses["coord_vis"] = loss_coord_vis
                if self.lambda_coord_occluded > 0:
                    raw_losses["coord_occ"] = loss_coord_occ

        if self.lambda_anatomical > 0:
            curr_lambda = self._get_current_lambda_ana(self.current_epoch)
            if not self.using_model_coords:
                pred_coords = self.soft_argmax(outputs)
            loss_ana = self.anatomical_criterion(pred_coords)
            metrics["loss_ana"] = loss_ana.item()
            metrics["lambda_ana"] = curr_lambda

            if not self.use_uncertainty:
                loss = loss + curr_lambda * loss_ana
            else:
                raw_losses["ana"] = loss_ana

        if self.use_uncertainty:
            loss, weighted_metrics = self.uncertainty_loss(raw_losses)
            metrics.update(weighted_metrics)

        metrics["loss"] = loss.item()

        # Log validation metrics prefixed with val_
        for k, v in metrics.items():
            self.log(
                f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=False
            )

        return loss

    def configure_optimizers(self):
        from src.training.factory import build_optimizer

        # We pass self as the trainer/mock-trainer to retain full factory compatibility
        optimizer = build_optimizer(self.model, self, self.config)
        return optimizer
