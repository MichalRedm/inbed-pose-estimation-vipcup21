"""
Implements the training logic for Contrastive Unpaired Translation (CUT).
Provides both the PyTorch Lightning module and the Trainer wrapper.
"""
import torch
import torch.nn as nn
import itertools
import pytorch_lightning as pl
from typing import Dict, Any, cast

from .base_trainer import BaseTrainer
from src.models.cyclegan.generator import GeneratorResNet
from src.models.cyclegan.discriminator import Discriminator
from src.models.cyclegan.loss import GANLoss
from src.models.cyclegan.cut_loss import PatchSampleF, PatchNCELoss

class CUTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Contrastive Unpaired Translation (CUT).

    Handles the generator and discriminator updates, including the PatchNCE
    contrastive loss and optional pose-preservation loss.
    """
    def __init__(
        self,
        G: nn.Module,
        D: nn.Module,
        F: nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        config: Dict[str, Any],
        pose_estimator: nn.Module = None,
    ):
        super().__init__()
        self.G = G
        self.D = D
        self.F_net = F
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.config = config
        self.P = pose_estimator

        train_cfg = config.get("training", {})
        self.lambda_nce = train_cfg.get("lambda_nce", 1.0)
        self.lambda_gan = train_cfg.get("lambda_gan", 1.0)
        self.lambda_pose = train_cfg.get("lambda_pose", 0.0)
        self.num_patches = train_cfg.get("num_patches", 256)
        self.use_nce_idt = train_cfg.get("use_nce_idt", False)
        self.nce_layers = train_cfg.get("nce_layers", [0, 1, 2, 3, 4])

        self.criterion_GAN = GANLoss()
        self.criterion_NCE = PatchNCELoss()
        self.criterion_pose = nn.MSELoss()

        use_amp = torch.cuda.is_available()
        self.scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """
        Passes the input through the generator.

        Args:
            x: The input image tensor.
            **kwargs: Additional arguments for the generator.

        Returns:
            The generated image tensor.
        """
        return self.G(x, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        if batch is None:
            return

        if self._trainer is not None and hasattr(self._trainer, "strategy"):
            opt_g, opt_d = cast(Any, self.optimizers())
        else:
            opt_g, opt_d = self.optimizer_G, self.optimizer_D
            
        real_A, real_B = batch
        device_type = self.device.type
        use_amp = device_type == "cuda"

        # Generator Update
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            fake_B, feat_k = self.G(real_A, return_features=True, nce_layers=self.nce_layers)
            feat_k = [f.detach() for f in feat_k]
            feat_q = self.G(fake_B, encode_only=True, nce_layers=self.nce_layers)

            loss_G_GAN = self.criterion_GAN(self.D(fake_B), True) * self.lambda_gan

            pool_q, patch_ids = self.F_net(feat_q, num_patches=self.num_patches)
            pool_k, _ = self.F_net(feat_k, patch_ids=patch_ids)
            loss_NCE = self.criterion_NCE(pool_q, pool_k) * self.lambda_nce

            if self.use_nce_idt:
                idt_B, feat_k_idt = self.G(real_B, return_features=True, nce_layers=self.nce_layers)
                feat_k_idt = [f.detach() for f in feat_k_idt]
                feat_q_idt = self.G(idt_B, encode_only=True, nce_layers=self.nce_layers)
                pool_q_idt, patch_ids_idt = self.F_net(feat_q_idt, num_patches=self.num_patches)
                pool_k_idt, _ = self.F_net(feat_k_idt, patch_ids=patch_ids_idt)
                loss_NCE_idt = self.criterion_NCE(pool_q_idt, pool_k_idt) * self.lambda_nce
            else:
                loss_NCE_idt = torch.tensor(0.0, device=self.device)

            loss_G = loss_G_GAN + loss_NCE + loss_NCE_idt

            if self.P is not None and self.lambda_pose > 0:
                with torch.no_grad():
                    pose_real = self.P(real_A)
                pose_fake = self.P(fake_B)
                loss_pose = self.criterion_pose(pose_fake, pose_real) * self.lambda_pose
                loss_G += loss_pose
            else:
                loss_pose = torch.tensor(0.0, device=self.device)

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(cast(Any, opt_g))
        self.scaler_G.update()

        # Discriminator Update
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            loss_D_real = self.criterion_GAN(self.D(real_B), True)
            loss_D_fake = self.criterion_GAN(self.D(fake_B.detach()), False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.step(cast(Any, opt_d))
        self.scaler_D.update()

        metrics = {
            "loss": loss_G.item(),
            "adv_loss": loss_G_GAN.item(),
            "nce_loss": loss_NCE.item(),
            "nce_idt_loss": loss_NCE_idt.item(),
            "pose_loss": loss_pose.item() if self.P is not None else 0.0,
            "d_loss": loss_D.item(),
        }
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        real_A, real_B = batch
        device_type = self.device.type
        use_amp = device_type == "cuda"
        
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            fake_B, feat_k = self.G(real_A, return_features=True, nce_layers=self.nce_layers)
            feat_k = [f.detach() for f in feat_k]
            feat_q = self.G(fake_B, encode_only=True, nce_layers=self.nce_layers)

            loss_G_GAN = self.criterion_GAN(self.D(fake_B), True) * self.lambda_gan

            pool_q, patch_ids = self.F_net(feat_q, num_patches=self.num_patches)
            pool_k, _ = self.F_net(feat_k, patch_ids=patch_ids)
            loss_NCE = self.criterion_NCE(pool_q, pool_k) * self.lambda_nce

            if self.use_nce_idt:
                idt_B, feat_k_idt = self.G(real_B, return_features=True, nce_layers=self.nce_layers)
                feat_k_idt = [f.detach() for f in feat_k_idt]
                feat_q_idt = self.G(idt_B, encode_only=True, nce_layers=self.nce_layers)
                pool_q_idt, patch_ids_idt = self.F_net(feat_q_idt, num_patches=self.num_patches)
                pool_k_idt, _ = self.F_net(feat_k_idt, patch_ids=patch_ids_idt)
                loss_NCE_idt = self.criterion_NCE(pool_q_idt, pool_k_idt) * self.lambda_nce
            else:
                loss_NCE_idt = torch.tensor(0.0, device=self.device)

            loss_G = loss_G_GAN + loss_NCE + loss_NCE_idt

            if self.P is not None and self.lambda_pose > 0:
                with torch.no_grad():
                    pose_real = self.P(real_A)
                pose_fake = self.P(fake_B)
                loss_pose = self.criterion_pose(pose_fake, pose_real) * self.lambda_pose
                loss_G += loss_pose
            else:
                loss_pose = torch.tensor(0.0, device=self.device)

        metrics = {
            "loss": loss_G.item(),
            "adv_loss": loss_G_GAN.item(),
            "nce_loss": loss_NCE.item(),
            "nce_idt_loss": loss_NCE_idt.item(),
            "pose_loss": loss_pose.item() if self.P is not None else 0.0,
        }
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return cast(torch.Tensor, loss_G)

    def configure_optimizers(self) -> Any:
        return [self.optimizer_G, self.optimizer_D]


class CUTTrainer(BaseTrainer):
    """
    Trainer wrapper for CUT using PyTorch Lightning.
    
    Orchestrates the setup of the models, optimizers, and the Lightning trainer
    for Contrastive Unpaired Translation.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        input_shape = (3, 256, 256)
        pretrained = config.get("training", {}).get("pretrained_gan", True)

        self.G = GeneratorResNet(
            input_shape, num_residual_blocks=9, pretrained=pretrained
        ).to(device)
        self.D = Discriminator(input_shape).to(device)

        self.nce_layers = config.get("training", {}).get("nce_layers", [0, 1, 2, 3, 4])
        default_channels = [128, 256, 256, 256, 256]
        in_channels_list = [default_channels[i] for i in self.nce_layers]

        self.F_net = PatchSampleF(in_channels_list=in_channels_list).to(device)

        super().__init__(self.G, config, device, rank, world_size)

        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("lr", 0.0002)
        self.b1 = train_cfg.get("b1", 0.5)
        self.b2 = train_cfg.get("b2", 0.999)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G.parameters(), self.F_net.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        
        self.pose_estimator = None
        pretrained_pose = config.get("training", {}).get("pretrained_pose_estimator_path", None)
        if pretrained_pose:
            from src.models import build_model
            if self.is_main:
                print(f"[CUTTrainer] Loading frozen pose estimator from {pretrained_pose}")
            ckpt = torch.load(pretrained_pose, map_location="cpu", weights_only=False)
            
            if "config" in ckpt:
                pose_model = build_model(ckpt["config"]).to(device)
            else:
                pose_model = build_model(config).to(device)
                
            if "model_state_dict" in ckpt:
                pose_model.load_state_dict(ckpt["model_state_dict"])
            else:
                pose_model.load_state_dict(ckpt)
                
            pose_model.eval()
            for param in pose_model.parameters():
                param.requires_grad = False
            self.pose_estimator = pose_model

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        from .lightning_callbacks import DashboardTelemetryCallback

        lightning_module = CUTLightningModule(
            G=self.G,
            D=self.D,
            F=self.F_net,
            optimizer_G=self.optimizer_G,
            optimizer_D=self.optimizer_D,
            config=self.config,
            pose_estimator=self.pose_estimator,
        )

        callbacks = [DashboardTelemetryCallback(self)]
        trainer = self._setup_pl_trainer(callbacks=callbacks, use_ddp=False)

        if self.is_main:
            print("[CUTTrainer] Starting PyTorch Lightning training loop...")

        if self.resume_state:
            self._load_extra_checkpoint_data(self.resume_state)

        self._run_pl_fit(trainer, lightning_module, train_loader, val_loader)

    def _load_extra_checkpoint_data(self, state: Dict[str, Any]) -> None:
        if "D_state_dict" in state:
            self.D.load_state_dict(state["D_state_dict"])
        if "F_state_dict" in state:
            self.F_net.load_state_dict(state["F_state_dict"])
        if "optimizer_G_state_dict" in state:
            self.optimizer_G.load_state_dict(state["optimizer_G_state_dict"])
        if "optimizer_D_state_dict" in state:
            self.optimizer_D.load_state_dict(state["optimizer_D_state_dict"])

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "D_state_dict": self.D.state_dict(),
            "F_state_dict": self.F_net.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        }
