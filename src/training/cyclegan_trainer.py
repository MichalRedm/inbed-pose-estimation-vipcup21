import torch
import torch.nn as nn
import itertools
from typing import Dict, Any
from .base_trainer import BaseTrainer
from src.models.cyclegan import GeneratorResNet, Discriminator, GANLoss


class CycleGANTrainer(BaseTrainer):
    """
    Trainer for CycleGAN domain translation.
    Translates between Domain A (Uncovered) and Domain B (Covered).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        # In CycleGAN, we have 4 models. We pass G_AB as the "primary" model to BaseTrainer
        # although we will handle all 4 manually.
        input_shape = (3, 256, 256)
        # Use pretrained weights if specified in config (default True based on ideas_log)
        pretrained = config.get("training", {}).get("pretrained_gan", True)
        self.G_AB = GeneratorResNet(input_shape, num_residual_blocks=6, pretrained=pretrained).to(device)
        self.G_BA = GeneratorResNet(input_shape, num_residual_blocks=6, pretrained=pretrained).to(device)
        self.D_A = Discriminator(input_shape).to(device)
        self.D_B = Discriminator(input_shape).to(device)

        super().__init__(self.G_AB, config, device, rank, world_size)

        # Hyperparameters
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("lr", 0.0002)
        self.b1 = train_cfg.get("b1", 0.5)
        self.b2 = train_cfg.get("b2", 0.999)
        self.lambda_cyc = train_cfg.get("lambda_cycle", 10.0)
        self.lambda_id = train_cfg.get("lambda_identity", 5.0)

        # Losses
        self.criterion_GAN = GANLoss().to(device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # CycleGAN expects a tuple of (Real_A, Real_B)
        # We assume the dataloader yields this.
        real_A, real_B = batch
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # ------------------
        #  Train Generators
        # ------------------
        self.optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.G_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), True)

        fake_A = self.G_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), True)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)

        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = (
            loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
        )
        loss_G.backward()
        self.optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        self.optimizer_D_A.zero_grad()
        loss_real = self.criterion_GAN(self.D_A(real_A), True)
        loss_fake = self.criterion_GAN(self.D_A(fake_A.detach()), False)
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        self.optimizer_D_B.zero_grad()
        loss_real = self.criterion_GAN(self.D_B(real_B), True)
        loss_fake = self.criterion_GAN(self.D_B(fake_B.detach()), False)
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        self.optimizer_D_B.step()

        return {
            "loss": loss_G.item(),
            "adv_loss": loss_GAN.item(),
            "cycle_loss": loss_cycle.item(),
            "id_loss": loss_identity.item(),
            "d_loss": (loss_D_A + loss_D_B).item(),
        }

    def _val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # For CycleGAN, validation is just tracking the same losses on val set
        # but without backward pass.
        with torch.no_grad():
            return self._train_step(batch)

    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                # Rename keys for history
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            if self.is_main:
                epoch_data = {"epoch": epoch + 1}
                epoch_data.update(train_metrics)
                epoch_data.update(val_metrics)
                self.update_history(epoch_data)

                # Checkpointing
                # For CycleGAN, we save based on loss (lower is better)
                val_loss = val_metrics.get("val_loss", train_metrics["loss"])
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                self.save_checkpoint(f"epoch_{epoch + 1}", is_best=is_best)

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
        }

    def get_display_metadata(self) -> Dict[str, Any]:
        return {
            "loss_labels": {
                "loss": "Generator Loss",
                "adv_loss": "Adversarial",
                "cycle_loss": "Cycle Consistency",
                "id_loss": "Identity",
                "d_loss": "Discriminator",
            },
            "primary_metric": "loss",
        }
