import torch
import torch.nn as nn
import itertools
from typing import Dict, Any
from .base_trainer import BaseTrainer
from src.models.cyclegan.generator import GeneratorResNet
from src.models.cyclegan.discriminator import Discriminator
from src.models.cyclegan.loss import GANLoss
from src.models.cyclegan.cut_loss import PatchSampleF, PatchNCELoss


class CUTTrainer(BaseTrainer):
    """
    Trainer for Contrastive Unpaired Translation (CUT).
    Translates Domain A to Domain B without a cycle-consistency constraint,
    utilizing Patchwise Contrastive Learning (InfoNCE).
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
        
        # Core Models
        self.G = GeneratorResNet(
            input_shape, num_residual_blocks=9, pretrained=pretrained
        ).to(device)
        self.D = Discriminator(input_shape).to(device)
        self.F = PatchSampleF().to(device)

        super().__init__(self.G, config, device, rank, world_size)

        # Hyperparameters
        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("lr", 0.0002)
        self.b1 = train_cfg.get("b1", 0.5)
        self.b2 = train_cfg.get("b2", 0.999)
        self.lambda_nce = train_cfg.get("lambda_nce", 1.0)
        self.lambda_gan = train_cfg.get("lambda_gan", 1.0)
        self.num_patches = train_cfg.get("num_patches", 256)

        # Losses
        self.criterion_GAN = GANLoss().to(device)
        self.criterion_NCE = PatchNCELoss().to(device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G.parameters(), self.F.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        # GradScalers for mixed precision
        use_amp = self.device.type == "cuda"
        self.scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _calculate_losses(self, batch: Any) -> Dict[str, torch.Tensor]:
        real_A, real_B = batch
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # ------------------
        #  Generator
        # ------------------
        # G(A) -> B
        fake_B, feat_k = self.G(real_A, return_features=True)
        # We detach feat_k to prevent gradients flowing into the encoder for the target
        feat_k = [f.detach() for f in feat_k]
        
        # Features of generated fake_B
        _, feat_q = self.G(fake_B, return_features=True)

        # GAN Loss
        pred_fake = self.D(fake_B)
        loss_G_GAN = self.criterion_GAN(pred_fake, True) * self.lambda_gan

        # NCE Loss for translation
        pool_q, patch_ids = self.F(feat_q, num_patches=self.num_patches)
        pool_k, _ = self.F(feat_k, patch_ids=patch_ids)
        loss_NCE = self.criterion_NCE(pool_q, pool_k) * self.lambda_nce

        # Identity NCE Loss (Domain B -> Domain B)
        idt_B, feat_k_idt = self.G(real_B, return_features=True)
        feat_k_idt = [f.detach() for f in feat_k_idt]
        _, feat_q_idt = self.G(idt_B, return_features=True)

        pool_q_idt, patch_ids_idt = self.F(feat_q_idt, num_patches=self.num_patches)
        pool_k_idt, _ = self.F(feat_k_idt, patch_ids=patch_ids_idt)
        loss_NCE_idt = self.criterion_NCE(pool_q_idt, pool_k_idt) * self.lambda_nce

        loss_G = loss_G_GAN + loss_NCE + loss_NCE_idt

        # ------------------
        #  Discriminator
        # ------------------
        pred_real = self.D(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True)
        
        pred_fake_detach = self.D(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake_detach, False)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return {
            "loss": loss_G,
            "adv_loss": loss_G_GAN,
            "nce_loss": loss_NCE,
            "nce_idt_loss": loss_NCE_idt,
            "d_loss": loss_D,
        }

    def _train_step(self, batch: Any) -> Dict[str, float]:
        device_type = self.device.type
        use_amp = device_type == "cuda"
        
        real_A, real_B = batch
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # ------------------
        #  Generator Update
        # ------------------
        self.optimizer_G.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            fake_B, feat_k = self.G(real_A, return_features=True)
            feat_k = [f.detach() for f in feat_k]
            _, feat_q = self.G(fake_B, return_features=True)

            loss_G_GAN = self.criterion_GAN(self.D(fake_B), True) * self.lambda_gan

            pool_q, patch_ids = self.F(feat_q, num_patches=self.num_patches)
            pool_k, _ = self.F(feat_k, patch_ids=patch_ids)
            loss_NCE = self.criterion_NCE(pool_q, pool_k) * self.lambda_nce

            idt_B, feat_k_idt = self.G(real_B, return_features=True)
            feat_k_idt = [f.detach() for f in feat_k_idt]
            _, feat_q_idt = self.G(idt_B, return_features=True)

            pool_q_idt, patch_ids_idt = self.F(feat_q_idt, num_patches=self.num_patches)
            pool_k_idt, _ = self.F(feat_k_idt, patch_ids=patch_ids_idt)
            loss_NCE_idt = self.criterion_NCE(pool_q_idt, pool_k_idt) * self.lambda_nce

            loss_G = loss_G_GAN + loss_NCE + loss_NCE_idt

        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

        # -----------------------
        #  Discriminator Update
        # -----------------------
        self.optimizer_D.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            loss_D_real = self.criterion_GAN(self.D(real_B), True)
            loss_D_fake = self.criterion_GAN(self.D(fake_B.detach()), False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        self.scaler_D.scale(loss_D).backward()
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()

        # Compile metrics dict
        losses = {
            "loss": loss_G,
            "adv_loss": loss_G_GAN,
            "nce_loss": loss_NCE,
            "nce_idt_loss": loss_NCE_idt,
            "d_loss": loss_D,
        }
        return {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}

    def _val_step(self, batch: Any) -> Dict[str, float]:
        device_type = self.device.type
        use_amp = device_type == "cuda"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            losses = self._calculate_losses(batch)
        return {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}

    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            if self.is_main:
                epoch_data = {"epoch": epoch + 1}
                epoch_data.update(train_metrics)
                epoch_data.update(val_metrics)
                self.update_history(epoch_data)

                # Checkpointing
                val_loss = val_metrics.get("val_loss", train_metrics["loss"])
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                self.save_checkpoint(f"epoch_{epoch + 1}", is_best=is_best)

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "D_state_dict": self.D.state_dict(),
            "F_state_dict": self.F.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        }
