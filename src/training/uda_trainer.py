import torch
import torch.nn as nn
from typing import Dict, Any, Optional
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
        from .lightning_module import UDALightningModule
        from .lightning_callbacks import DashboardTelemetryCallback

        # 1. Instantiate Lightning Module
        lightning_module = UDALightningModule(
            model=self.model,
            discriminator=self.discriminator,
            optimizer=self.optimizer,
            optimizer_d=self.optimizer_d,
            criterion=self.criterion,
            config=self.config,
        )

        # 2. Instantiate custom callbacks
        callbacks = [DashboardTelemetryCallback(self)]

        # 3. Configure Trainer options
        trainer = self._setup_pl_trainer(callbacks=callbacks)

        if self.is_main:
            print("[UDATrainer] Starting refactored PyTorch Lightning training loop...")
            print(
                f"[UDATrainer] Accelerator: {trainer.accelerator}, Devices: {trainer.num_devices}, Strategy: {trainer.strategy}"
            )

        self._run_pl_fit(trainer, lightning_module, train_loader, val_loader)
