import torch
import torch.nn as nn
import itertools
from typing import Dict, Any, cast
from .base_trainer import BaseTrainer
from src.models.cyclegan import GeneratorResNet, Discriminator, GANLoss


class CycleGANTrainer(BaseTrainer):
    """
    Trainer for CycleGAN domain translation.
    Translates between Domain A (Uncovered) and Domain B (Covered).
    """

    G_AB: nn.Module
    G_BA: nn.Module
    D_A: nn.Module
    D_B: nn.Module
    lr: float
    b1: float
    b2: float
    lambda_cyc: float
    lambda_id: float
    criterion_GAN: GANLoss
    criterion_cycle: nn.Module
    criterion_identity: nn.Module
    optimizer_G: torch.optim.Optimizer
    optimizer_D_A: torch.optim.Optimizer
    optimizer_D_B: torch.optim.Optimizer
    scaler_G: torch.amp.GradScaler
    scaler_D_A: torch.amp.GradScaler
    scaler_D_B: torch.amp.GradScaler

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        # In CycleGAN, we have 4 models. We pass G_AB as the "primary" model to BaseTrainer
        # although we will handle all 4 manually.
        input_shape = (3, 256, 256)
        # Use pretrained weights if specified in config (default True based on ideas_log)
        pretrained = bool(config.get("training", {}).get("pretrained_gan", True))
        self.G_AB = GeneratorResNet(
            input_shape, num_residual_blocks=6, pretrained=pretrained
        ).to(device)
        self.G_BA = GeneratorResNet(
            input_shape, num_residual_blocks=6, pretrained=pretrained
        ).to(device)
        self.D_A = Discriminator(input_shape).to(device)
        self.D_B = Discriminator(input_shape).to(device)

        super().__init__(self.G_AB, config, device, rank, world_size)

        # Hyperparameters
        train_cfg: Dict[str, Any] = config.get("training", {})
        self.lr = float(train_cfg.get("lr", 0.0002))
        self.b1 = float(train_cfg.get("b1", 0.5))
        self.b2 = float(train_cfg.get("b2", 0.999))
        self.lambda_cyc = float(train_cfg.get("lambda_cycle", 10.0))
        self.lambda_id = float(train_cfg.get("lambda_identity", 5.0))

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

        # GradScalers for mixed precision (enabled only if device is CUDA)
        use_amp = self.device.type == "cuda"
        self.scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D_A = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D_B = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Model compilation (optional, enabled if compile is specified in config and PyTorch 2.x compile is available)
        compile_cfg = bool(train_cfg.get("compile", False))
        if compile_cfg and hasattr(torch, "compile"):
            if self.is_main:
                print("[CycleGANTrainer] Compiling models...")
            self.G_AB = cast(nn.Module, torch.compile(self.G_AB))
            self.G_BA = cast(nn.Module, torch.compile(self.G_BA))
            self.D_A = cast(nn.Module, torch.compile(self.D_A))
            self.D_B = cast(nn.Module, torch.compile(self.D_B))

    def _calculate_losses(self, batch: Any) -> Dict[str, torch.Tensor]:
        real_A, real_B = batch
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # ------------------
        #  Generators
        # ------------------
        # Identity loss (skip if lambda_identity <= 0)
        if self.lambda_id > 0:
            fake_B_id = self.G_AB(real_B)
            loss_id_B = self.criterion_identity(fake_B_id, real_B)
            fake_A_id = self.G_BA(real_A)
            loss_id_A = self.criterion_identity(fake_A_id, real_A)
            loss_identity = (loss_id_A + loss_id_B) / 2
        else:
            loss_identity = torch.tensor(0.0, device=self.device)

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

        # Total Generator Loss
        loss_G = (
            loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
        )

        # -----------------------
        #  Discriminators
        # -----------------------
        # Discriminator A
        loss_real_A = self.criterion_GAN(self.D_A(real_A), True)
        loss_fake_A = self.criterion_GAN(self.D_A(fake_A.detach()), False)
        loss_D_A = (loss_real_A + loss_fake_A) / 2

        # Discriminator B
        loss_real_B = self.criterion_GAN(self.D_B(real_B), True)
        loss_fake_B = self.criterion_GAN(self.D_B(fake_B.detach()), False)
        loss_D_B = (loss_real_B + loss_fake_B) / 2

        return {
            "loss": loss_G,
            "adv_loss": loss_GAN,
            "cycle_loss": loss_cycle,
            "id_loss": loss_identity,
            "loss_D_A": loss_D_A,
            "loss_D_B": loss_D_B,
            "d_loss": loss_D_A + loss_D_B,
        }

    def _train_step(self, batch: Any) -> Dict[str, float]:
        real_A, real_B = batch
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        device_type = self.device.type
        use_amp = device_type == "cuda"

        # ------------------
        #  Generators Update
        # ------------------
        self.optimizer_G.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            # Identity loss (skip if lambda_identity <= 0)
            if self.lambda_id > 0:
                fake_B_id = self.G_AB(real_B)
                loss_id_B = self.criterion_identity(fake_B_id, real_B)
                fake_A_id = self.G_BA(real_A)
                loss_id_A = self.criterion_identity(fake_A_id, real_A)
                loss_identity = (loss_id_A + loss_id_B) / 2
            else:
                loss_identity = torch.tensor(0.0, device=self.device)

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

            # Total Generator Loss
            loss_G = (
                loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
            )

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

        # -----------------------
        #  Discriminators Update
        # -----------------------
        self.optimizer_D_A.zero_grad(set_to_none=True)
        self.optimizer_D_B.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            # Discriminator A
            loss_real_A = self.criterion_GAN(self.D_A(real_A), True)
            loss_fake_A = self.criterion_GAN(self.D_A(fake_A.detach()), False)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

        self.scaler_D_A.scale(loss_D_A).backward()
        self.scaler_D_A.step(self.optimizer_D_A)
        self.scaler_D_A.update()

        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            # Discriminator B
            loss_real_B = self.criterion_GAN(self.D_B(real_B), True)
            loss_fake_B = self.criterion_GAN(self.D_B(fake_B.detach()), False)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

        self.scaler_D_B.scale(loss_D_B).backward()
        self.scaler_D_B.step(self.optimizer_D_B)
        self.scaler_D_B.update()

        # Compile metrics dict
        losses = {
            "loss": loss_G,
            "adv_loss": loss_GAN,
            "cycle_loss": loss_cycle,
            "id_loss": loss_identity,
            "loss_D_A": loss_D_A,
            "loss_D_B": loss_D_B,
            "d_loss": loss_D_A + loss_D_B,
        }
        return {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}

    def _val_step(self, batch: Any) -> Dict[str, float]:
        device_type = self.device.type
        use_amp = device_type == "cuda"
        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            losses = self._calculate_losses(batch)
        return {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        from .lightning_module import CycleGANLightningModule
        from .lightning_callbacks import DashboardTelemetryCallback

        # 1. Instantiate Lightning Module
        lightning_module = CycleGANLightningModule(
            G_AB=self.G_AB,
            G_BA=self.G_BA,
            D_A=self.D_A,
            D_B=self.D_B,
            optimizer_G=self.optimizer_G,
            optimizer_D_A=self.optimizer_D_A,
            optimizer_D_B=self.optimizer_D_B,
            config=self.config,
        )

        # 2. Instantiate custom callbacks
        callbacks = [DashboardTelemetryCallback(self)]

        # 3. Configure Trainer options (No DDP for CycleGAN yet)
        trainer = self._setup_pl_trainer(callbacks=callbacks, use_ddp=False)

        if self.is_main:
            print(
                "[CycleGANTrainer] Starting refactored PyTorch Lightning training loop..."
            )
            print(
                f"[CycleGANTrainer] Accelerator: {trainer.accelerator}, Devices: {trainer.num_devices}, Strategy: {trainer.strategy}"
            )

        self._run_pl_fit(trainer, lightning_module, train_loader, val_loader)

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
        }
