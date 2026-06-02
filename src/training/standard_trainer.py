import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from .losses import AnatomicalLoss, UncertaintyWeighting
from ..models.layers import SoftArgmax2D


def generate_pytorch_heatmaps(
    joints: torch.Tensor,
    heatmap_size: Tuple[int, int] = (64, 64),
    image_size: Tuple[int, int] = (256, 256),
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Generate 2D Gaussian heatmaps on PyTorch tensors directly on the target device.
    joints: tensor of shape (B, 3, 14) -> (coords, joints) -> joints[:, :2, :] is (x, y)
    heatmap_size: (H_out, W_out)
    image_size: (H_in, W_in)
    sigma: float
    """
    B, _, J = joints.shape
    H_out, W_out = heatmap_size
    H_in, W_in = image_size
    device = joints.device

    # Scale joints to heatmap size
    scale_x = W_out / W_in
    scale_y = H_out / H_in

    mu_x = joints[:, 0, :] * scale_x  # (B, J)
    mu_y = joints[:, 1, :] * scale_y  # (B, J)
    visibility = joints[:, 2, :]  # (B, J)

    # Create coordinate grids
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_out, device=device, dtype=torch.float32),
        torch.arange(W_out, device=device, dtype=torch.float32),
        indexing="ij",
    )  # (H_out, W_out)

    grid_x = grid_x.view(1, 1, H_out, W_out)  # (1, 1, H_out, W_out)
    grid_y = grid_y.view(1, 1, H_out, W_out)  # (1, 1, H_out, W_out)

    mu_x = mu_x.view(B, J, 1, 1)  # (B, J, 1, 1)
    mu_y = mu_y.view(B, J, 1, 1)  # (B, J, 1, 1)

    # Generate Gaussian
    dist_sq = (grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2
    sigma_val = float(sigma)
    heatmaps = torch.exp(-dist_sq / (2 * sigma_val**2))

    # Apply visibility and out-of-bounds mask
    invalid_mask = (visibility > 1) | (
        (joints[:, 0, :] == 0) & (joints[:, 1, :] == 0)
    )  # (B, J)
    invalid_mask = invalid_mask.view(B, J, 1, 1)

    heatmaps = heatmaps.masked_fill(invalid_mask, 0.0)

    # Also mask out individual joints that are out of bounds
    out_of_bounds = (
        (mu_x < 0) | (mu_y < 0) | (mu_x >= W_out) | (mu_y >= H_out)
    )  # (B, J, 1, 1)
    heatmaps = heatmaps.masked_fill(out_of_bounds, 0.0)

    return heatmaps


class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer for pose estimation with optional anatomical constraints.
    """

    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    unfreeze_epoch: Optional[int]
    backbone_lr_ratio: float
    lambda_anatomical: float
    warmup_epochs: int
    anatomical_mode: str
    lambda_coord: float
    lambda_coord_occluded: float
    sigma_start: float
    sigma_end: float
    soft_argmax: SoftArgmax2D
    anatomical_criterion: AnatomicalLoss
    use_uncertainty: bool
    tasks: List[str]
    uncertainty_loss: UncertaintyWeighting

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__(model, config, device, rank, world_size)
        if optimizer is not None:
            self.optimizer = optimizer
        self.criterion = criterion

        # Anatomical constraints setup
        training_cfg: Dict[str, Any] = config.get("training", {})
        self.unfreeze_epoch = training_cfg.get("unfreeze_epoch")
        self.backbone_lr_ratio = float(training_cfg.get("backbone_lr_ratio", 1.0))
        self.lambda_anatomical = float(training_cfg.get("lambda_anatomical", 0.0))
        self.warmup_epochs = int(training_cfg.get("warmup_epochs", 10))
        self.anatomical_mode = str(training_cfg.get("anatomical_mode", "hinge"))
        self.lambda_coord = float(training_cfg.get("lambda_coord", 0.0))
        self.lambda_coord_occluded = float(
            training_cfg.get("lambda_coord_occluded", 0.0)
        )
        self.sigma_start = float(training_cfg.get("sigma_start", 2.0))
        self.sigma_end = float(training_cfg.get("sigma_end", 2.0))

        if (
            self.lambda_anatomical > 0
            or self.lambda_coord > 0
            or self.lambda_coord_occluded > 0
        ):
            # Use high temperature to ensure soft-argmax focuses on the actual peak
            self.soft_argmax = SoftArgmax2D(temperature=100.0).to(device)

        if self.lambda_anatomical > 0:
            self.anatomical_criterion = AnatomicalLoss(
                device=str(device), mode=self.anatomical_mode
            ).to(device)

        # Multi-task uncertainty weighting
        self.use_uncertainty = bool(
            training_cfg.get("use_uncertainty_weighting", False)
        )
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
        num_epochs: int = int(self.config.get("training", {}).get("epochs", 30))
        if num_epochs <= 1:
            return self.sigma_start

        # Linear decay
        progress = min(
            epoch / (num_epochs * 0.7), 1.0
        )  # Reach sigma_end at 70% of training
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress

    def train_epoch(self, dataloader: Any, epoch: int) -> Dict[str, float]:
        self.current_epoch = epoch  # Store current epoch for steps

        # Dynamic Sigma Scheduling (moved to Dataset)
        if hasattr(dataloader.dataset, "set_sigma"):
            sigma = self._get_current_sigma(epoch)
            dataloader.dataset.set_sigma(sigma)

        return super().train_epoch(dataloader, epoch)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        joints = batch["joints"].to(self.device)  # (B, 3, 14)

        # Track current sigma for metrics and dynamic curriculum
        sigma = self._get_current_sigma(self.current_epoch)

        # Generate target heatmaps dynamically with high precision on the GPU!
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

        sigma = self._get_current_sigma(self.current_epoch)

        # Generate target heatmaps dynamically with high precision on the GPU!
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

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        from .lightning_module import PoseLightningModule
        from .lightning_callbacks import (
            DashboardTelemetryCallback,
            ProgressiveUnfreezingCallback,
        )

        # 1. Instantiate Lightning Module
        self.lightning_module = PoseLightningModule(
            model=self.model,
            config=self.config,
            criterion=self.criterion,
            optimizer=self.optimizer,
        )

        # Sync the unfreeze_epoch state if needed
        self.lightning_module.unfreeze_epoch = self.unfreeze_epoch

        # 2. Instantiate custom callbacks
        callbacks: List[Any] = [DashboardTelemetryCallback(self)]
        if self.unfreeze_epoch is not None:
            callbacks.append(ProgressiveUnfreezingCallback())

        # 3. Configure Trainer options
        trainer = self._setup_pl_trainer(callbacks=callbacks)

        # 4. Fit using PL Trainer
        if self.is_main:
            print(
                "[StandardTrainer] Starting refactored PyTorch Lightning training loop..."
            )
            print(
                f"[StandardTrainer] Accelerator: {trainer.accelerator}, Devices: {trainer.num_devices}, Strategy: {trainer.strategy}"
            )

        # 5. Restore state if resuming
        if self.resume_state:
            self._load_extra_checkpoint_data(self.resume_state)

        self._run_pl_fit(trainer, self.lightning_module, train_loader, val_loader)

    def _load_extra_checkpoint_data(self, state: Dict[str, Any]) -> None:
        if "optimizer_state_dict" in state:
            if self.is_main:
                print("[StandardTrainer] Restoring optimizer state from checkpoint.")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
