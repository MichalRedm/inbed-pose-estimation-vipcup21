import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_trainer import BaseTrainer


class UDATrainer(BaseTrainer):
    """
    Trainer for Unsupervised Domain Adaptation (UDA) using a Domain Discriminator
    and Gradient Reversal Layer (GRL).
    """

    optimizer: torch.optim.Optimizer
    discriminator: nn.Module
    optimizer_d: torch.optim.Optimizer
    criterion: nn.Module
    criterion_d: nn.Module
    lambda_adv: float
    warmup_epochs: int
    total_steps: int
    num_batches_per_epoch: int

    def __init__(
        self,
        model: nn.Module,
        discriminator: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__(model, config, device, rank, world_size)
        self.discriminator = discriminator.to(device)
        if optimizer is not None:
            self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.criterion = criterion
        self.criterion_d = nn.BCEWithLogitsLoss()

        # UDA specific params
        uda_cfg: Dict[str, Any] = config.get("uda", {})
        self.lambda_adv = float(uda_cfg.get("lambda_adv", 0.001))
        self.warmup_epochs = int(uda_cfg.get("warmup_epochs", 10))

        # Metrics tracking
        self.total_steps = 0

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 1. Prepare data
        img_target = batch["image"].to(self.device)  # Synthetically occluded (Target)
        img_source = batch["image_source"].to(self.device)  # Original clean (Source)
        target_heatmaps = batch["target"].to(self.device)  # Ground truth heatmaps

        B = img_target.size(0)

        # 2. Combined forward pass for efficiency and BN stability
        # We concatenate source and target images to process them in one go
        input_combined = torch.cat([img_source, img_target], dim=0)

        # Calculate GRL alpha (warmup)
        # alpha goes from 0 to 1 over warmup_epochs
        current_epoch = (
            self.total_steps / self.num_batches_per_epoch
            if hasattr(self, "num_batches_per_epoch")
            else 0.0
        )
        alpha = (
            min(1.0, current_epoch / self.warmup_epochs)
            if self.warmup_epochs > 0
            else 1.0
        )

        # HRNet forward with feature return
        # model expects (2B, C, H, W)
        outputs_combined, features_combined = self.model(
            input_combined, return_features=True
        )

        # Split outputs
        heatmaps_source = outputs_combined[:B]
        # heatmaps_target = outputs_combined[B:] # Not used for task loss as we only have labels for source

        # 3. Pose Task Loss (Source only)
        loss_pose = self.criterion(heatmaps_source, target_heatmaps)

        # 4. Domain Adversarial Loss
        # Features passed through GRL inside the discriminator
        d_out = self.discriminator(features_combined, alpha=alpha)

        # Domain labels: 0 for Source, 1 for Target
        domain_labels = torch.cat(
            [
                torch.zeros(B, 1, device=self.device),
                torch.ones(B, 1, device=self.device),
            ],
            dim=0,
        )

        loss_adv = self.criterion_d(d_out, domain_labels)

        # Total Loss
        loss_total = loss_pose + self.lambda_adv * loss_adv

        # 5. Optimization
        self.optimizer.zero_grad()
        self.optimizer_d.zero_grad()

        loss_total.backward()

        self.optimizer.step()
        self.optimizer_d.step()

        self.total_steps += 1

        return {
            "loss": loss_pose.item(),
            "adv_loss": loss_adv.item(),
            "total_loss": loss_total.item(),
            "alpha": alpha,
        }

    def _val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, targets)

        return {"loss": loss.item()}

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "total_steps": self.total_steps,
        }

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        self.num_batches_per_epoch = len(train_loader)

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics: Dict[str, float] = {}

            if val_loader:
                val_metrics = self.evaluate(val_loader)

            if self.is_main:
                # Log progress
                log_str = f"Epoch {epoch + 1}: "
                log_str += " ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
                if val_metrics:
                    log_str += " | " + " ".join(
                        [f"val_{k}={v:.4f}" for k, v in val_metrics.items()]
                    )
                # Stream comprehensive JSON summary to sidecar file
                summary_payload: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "progress": 1.0,
                    "is_summary": True,
                }
                summary_payload.update(train_metrics)
                if val_metrics:
                    summary_payload.update(
                        {f"val_{k}": v for k, v in val_metrics.items()}
                    )
                self._stream_metric(summary_payload)

                # Checkpointing
                val_loss = val_metrics.get("loss", float("inf"))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                self.save_checkpoint(f"epoch_{epoch + 1}", is_best=is_best)

                # History update
                epoch_data: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                self.update_history(epoch_data)
