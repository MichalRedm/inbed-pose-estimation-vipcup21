import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from .losses import AnatomicalLoss, UncertaintyWeighting
from ..models.layers import SoftArgmax2D


class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer for pose estimation with optional anatomical constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, config, device, rank, world_size)
        self.optimizer = optimizer
        self.criterion = criterion

        # Anatomical constraints setup
        training_cfg = config.get("training", {})
        self.lambda_anatomical = training_cfg.get("lambda_anatomical", 0.0)
        self.warmup_epochs = training_cfg.get("warmup_epochs", 10)
        self.anatomical_mode = training_cfg.get("anatomical_mode", "hinge")
        self.lambda_coord = training_cfg.get("lambda_coord", 0.0)
        self.lambda_coord_occluded = training_cfg.get("lambda_coord_occluded", 0.0)
        self.sigma_start = training_cfg.get("sigma_start", 2.0)
        self.sigma_end = training_cfg.get("sigma_end", 2.0)
        if (
            self.lambda_anatomical > 0
            or self.lambda_coord > 0
            or self.lambda_coord_occluded > 0
        ):
            # Use high temperature to ensure soft-argmax focuses on the actual peak
            self.soft_argmax = SoftArgmax2D(temperature=100.0).to(device)

        if self.lambda_anatomical > 0:
            self.anatomical_criterion = AnatomicalLoss(
                device=device, mode=self.anatomical_mode
            ).to(device)

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

            self.uncertainty_loss = UncertaintyWeighting(len(self.tasks)).to(device)
            if self.is_main:
                print(f"[Trainer] Using Uncertainty Weighting for tasks: {self.tasks}")

    def _get_current_lambda_ana(self, epoch: int) -> float:
        # Linear warmup over configurable epochs
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
        progress = min(
            epoch / (num_epochs * 0.7), 1.0
        )  # Reach sigma_end at 70% of training
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress

    def _generate_heatmaps_torch(
        self, joints: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """
        Generate Gaussian heatmaps in PyTorch.
        joints: (B, 3, 14) -> (x, y, vis) in image space (256x256)
        sigma: Gaussian spread
        Returns: (B, 14, 64, 64)
        """
        B, _, J = joints.shape
        H, W = self.heatmap_size
        device = joints.device

        # Grid of coordinates
        grid_y = torch.arange(H, device=device).float().view(1, 1, H, 1)
        grid_x = torch.arange(W, device=device).float().view(1, 1, 1, W)

        # Scale joints to heatmap resolution (256 -> 64)
        # Assuming image_size is (256, 256)
        mu_x = joints[:, 0, :].unsqueeze(-1).unsqueeze(-1) / 4.0
        mu_y = joints[:, 1, :].unsqueeze(-1).unsqueeze(-1) / 4.0
        vis = joints[:, 2, :].view(B, J, 1, 1)

        # Gaussian formula: exp(-((x-mu_x)^2 + (y-mu_y)^2) / (2 * sigma^2))
        dist_sq = (grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2
        heatmaps = torch.exp(-dist_sq / (2 * sigma**2))

        # Mask out missing joints (vis == 2)
        # Note: In VIP Cup, we supervise both visible (0) and occluded (1) with heatmaps
        heatmaps = heatmaps * (vis < 2).float()

        return heatmaps

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.current_epoch = epoch  # Store current epoch for steps
        return super().train_epoch(dataloader, epoch)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        joints = batch["joints"].to(self.device)  # (B, 3, 14)

        # Decide which targets to use
        if self.sigma_start != 2.0 or self.sigma_end != 2.0:
            sigma = self._get_current_sigma(self.current_epoch)
            targets = self._generate_heatmaps_torch(joints, sigma)
        else:
            targets = batch["target"].to(self.device)
            sigma = 2.0

        # Forward pass
        model_to_call = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if (
            hasattr(model_to_call, "forward")
            and "return_refined" in model_to_call.forward.__code__.co_varnames
        ):
            outputs, pred_coords = self.model(images, return_refined=True)
            using_model_coords = True
        else:
            outputs = self.model(images)
            using_model_coords = False

        loss_pose = self.criterion(outputs, targets)
        metrics = {"loss_pose": loss_pose.item(), "sigma": sigma}

        if self.use_uncertainty:
            raw_losses = {"pose": loss_pose}
        else:
            loss = loss_pose

        # 1. Coordinate regression loss
        if self.lambda_coord > 0 or self.lambda_coord_occluded > 0:
            if not using_model_coords:
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics

    def _val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        joints = batch["joints"].to(self.device)

        if self.sigma_start != 2.0 or self.sigma_end != 2.0:
            sigma = self._get_current_sigma(self.current_epoch)
            targets = self._generate_heatmaps_torch(joints, sigma)
        else:
            targets = batch["target"].to(self.device)
            sigma = 2.0

        # Forward pass
        model_to_call = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if (
            hasattr(model_to_call, "forward")
            and "return_refined" in model_to_call.forward.__code__.co_varnames
        ):
            outputs, pred_coords = self.model(images, return_refined=True)
            using_model_coords = True
        else:
            outputs = self.model(images)
            using_model_coords = False

        loss_pose = self.criterion(outputs, targets)
        metrics = {"loss_pose": loss_pose.item(), "sigma": sigma}

        if self.use_uncertainty:
            raw_losses = {"pose": loss_pose}
        else:
            loss = loss_pose

        if self.lambda_coord > 0 or self.lambda_coord_occluded > 0:
            if not using_model_coords:
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
        return metrics

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def fit(self, train_loader, val_loader=None):
        # Default to argmax for Heatmap models as it is more robust to background noise
        decode_method = "argmax"
        if self.is_main:
            print(
                f"[Trainer] Checkpoint saving criterion: best val PCK (decoder: {decode_method})"
            )

        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = {}
            val_pck = None

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_pck = self.compute_val_pck(val_loader, decode_method=decode_method)

            if self.is_main:
                log_str = f"Epoch {epoch + 1}: "
                log_str += " ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
                if val_metrics:
                    log_str += " | " + " ".join(
                        [f"val_{k}={v:.4f}" for k, v in val_metrics.items()]
                    )
                if val_pck is not None:
                    log_str += f" | val_pck={val_pck * 100:.2f}%"
                print(log_str)

                # is_best: PCK improved (primary) or loss improved if no PCK yet
                val_loss = val_metrics.get("loss", float("inf"))
                if val_pck is not None:
                    is_best = val_pck > self.best_val_pck
                    if is_best:
                        self.best_val_pck = val_pck
                else:
                    is_best = val_loss < self.best_val_loss

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                self.save_checkpoint(f"epoch_{epoch + 1}", is_best=is_best)

                epoch_data = {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                if val_pck is not None:
                    epoch_data["val_pck"] = val_pck
                self.update_history(epoch_data)
