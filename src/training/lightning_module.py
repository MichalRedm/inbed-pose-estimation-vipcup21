import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple, Union, cast

from src.training.losses import AnatomicalLoss, UncertaintyWeighting
from src.models.layers import SoftArgmax2D
from src.training.standard_trainer import generate_pytorch_heatmaps
from src.utils import decode_heatmaps


class PoseLightningModule(pl.LightningModule):
    """
    Standard PyTorch Lightning Module wrapping the pose estimation architectures (ViTPose, HRNet)
    and consolidating all the training steps, loss terms, and optimizer construction logic.
    """

    model: nn.Module
    config: Dict[str, Any]
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
    using_model_coords: bool
    soft_argmax: SoftArgmax2D
    anatomical_criterion: AnatomicalLoss
    use_uncertainty: bool
    tasks: List[str]
    uncertainty_loss: UncertaintyWeighting
    last_step_metrics: Dict[str, float]
    validation_step_outputs: List[Dict[str, torch.Tensor]]

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()

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

            self.uncertainty_loss = UncertaintyWeighting(len(self.tasks))

        self.validation_step_outputs = []

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return cast(
            Union[torch.Tensor, Tuple[torch.Tensor, ...]], self.model(x, **kwargs)
        )

    def _get_current_lambda_ana(self, epoch: int) -> float:
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
        progress = min(epoch / (num_epochs * 0.7), 1.0)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
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
        metrics: Dict[str, float] = {"loss_pose": loss_pose.item(), "sigma": sigma}

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
        return cast(torch.Tensor, loss)

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
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
        metrics: Dict[str, float] = {"loss_pose": loss_pose.item(), "sigma": sigma}

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

        # Decode and stash validation predictions for epoch-end PCK computation
        if getattr(model_to_call, "output_type", "heatmap") == "heatmap":
            method = self.config.get("training", {}).get("decode_method", "argmax")
            temp = float(
                self.config.get("training", {}).get("decode_temperature", 10.0)
            )
            preds = decode_heatmaps(outputs, (64, 64), method=method, temperature=temp)
        else:
            preds = outputs

        self.validation_step_outputs.append(
            {"preds": preds.detach().cpu(), "joints": joints.detach().cpu()}
        )

        return cast(torch.Tensor, loss)

    def configure_optimizers(self) -> Any:
        from src.training.factory import build_optimizer

        # We pass self as the trainer/mock-trainer to retain full factory compatibility
        optimizer = build_optimizer(self.model, self, self.config)
        return optimizer


class UDALightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module wrapping the Unsupervised Domain Adaptation (UDA) training loop
    incorporating a pose model and a domain discriminator via gradient reversal.
    """

    model: nn.Module
    discriminator: nn.Module
    config: Dict[str, Any]
    criterion: nn.Module
    criterion_d: nn.Module
    lambda_adv: float
    warmup_epochs: int
    total_steps: int
    last_step_metrics: Dict[str, float]
    opt_model: torch.optim.Optimizer
    opt_disc: torch.optim.Optimizer
    validation_step_outputs: List[Dict[str, torch.Tensor]]

    def __init__(
        self,
        model: nn.Module,
        discriminator: nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.discriminator = discriminator
        self.opt_model = optimizer
        self.opt_disc = optimizer_d
        self.criterion = criterion
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.config = config

        uda_cfg: Dict[str, Any] = config.get("uda", {})
        self.lambda_adv = float(uda_cfg.get("lambda_adv", 0.001))
        self.warmup_epochs = int(uda_cfg.get("warmup_epochs", 10))
        self.total_steps = 0
        self.last_step_metrics = {}
        self.validation_step_outputs = []

        # Manual optimization for multiple optimizers sequential training
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        return self.model(x, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        if batch is None:
            return

        if self._trainer is not None and hasattr(self._trainer, "strategy"):
            opt, opt_d = self.optimizers()
        else:
            opt, opt_d = self.opt_model, self.opt_disc

        img_target = batch["image"]
        img_source = batch["image_source"]
        target_heatmaps = batch["target"]
        B = img_target.size(0)

        # Combined forward pass for efficiency and BN stability
        input_combined = torch.cat([img_source, img_target], dim=0)

        # Calculate GRL alpha (warmup)
        if self._trainer is not None:
            num_batches = self.trainer.num_training_batches
            if num_batches <= 0 or num_batches == float("inf"):
                num_batches = (
                    len(self.trainer.train_dataloader)
                    if self.trainer.train_dataloader
                    else 1
                )
        else:
            num_batches = 1

        current_epoch = self.total_steps / max(num_batches, 1)
        alpha = (
            min(1.0, current_epoch / self.warmup_epochs)
            if self.warmup_epochs > 0
            else 1.0
        )

        # HRNet forward with feature return
        model_to_call = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        outputs_combined, features_combined = model_to_call(
            input_combined, return_features=True
        )

        # Split outputs
        heatmaps_source = outputs_combined[:B]

        # Pose Task Loss (Source only)
        loss_pose = self.criterion(heatmaps_source, target_heatmaps)

        # Domain Adversarial Loss
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

        # Optimization
        opt.zero_grad()
        opt_d.zero_grad()

        if self._trainer is not None and hasattr(self._trainer, "strategy"):
            self.manual_backward(loss_total)
        else:
            loss_total.backward()

        opt.step()
        opt_d.step()

        self.total_steps += 1

        metrics = {
            "loss": loss_pose.item(),
            "adv_loss": loss_adv.item(),
            "total_loss": loss_total.item(),
            "alpha": alpha,
        }
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.last_step_metrics = metrics

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["target"]

        outputs = self.model(images)
        loss = self.criterion(outputs, targets)

        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.validation_step_outputs.append({"loss": loss.detach().cpu()})
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def configure_optimizers(self) -> Any:
        return [self.opt_model, self.opt_disc]


class CycleGANLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module wrapping the CycleGAN domain translation training loop.
    """

    G_AB: nn.Module
    G_BA: nn.Module
    D_A: nn.Module
    D_B: nn.Module
    config: Dict[str, Any]
    optimizer_G: torch.optim.Optimizer
    optimizer_D_A: torch.optim.Optimizer
    optimizer_D_B: torch.optim.Optimizer
    lambda_cyc: float
    lambda_id: float
    criterion_GAN: nn.Module
    criterion_cycle: nn.Module
    criterion_identity: nn.Module
    scaler_G: torch.amp.GradScaler
    scaler_D_A: torch.amp.GradScaler
    scaler_D_B: torch.amp.GradScaler
    last_step_metrics: Dict[str, float]
    validation_step_outputs: List[Dict[str, torch.Tensor]]

    def __init__(
        self,
        G_AB: nn.Module,
        G_BA: nn.Module,
        D_A: nn.Module,
        D_B: nn.Module,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D_A: torch.optim.Optimizer,
        optimizer_D_B: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        self.config = config

        train_cfg: Dict[str, Any] = config.get("training", {})
        self.lambda_cyc = float(train_cfg.get("lambda_cycle", 10.0))
        self.lambda_id = float(train_cfg.get("lambda_identity", 5.0))

        # We construct losses on creation
        from src.models.cyclegan import GANLoss

        self.criterion_GAN = GANLoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Scalers for manual mixed precision
        use_amp = torch.cuda.is_available()
        self.scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D_A = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D_B = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.last_step_metrics = {}
        self.validation_step_outputs = []
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        return self.G_AB(x, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        if batch is None:
            return

        if self._trainer is not None and hasattr(self._trainer, "strategy"):
            opt_g, opt_da, opt_db = self.optimizers()
        else:
            opt_g, opt_da, opt_db = self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B
        real_A, real_B = batch

        device_type = self.device.type
        use_amp = device_type == "cuda"

        # Generators Update
        opt_g.zero_grad(set_to_none=True)

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
        self.scaler_G.step(opt_g)
        self.scaler_G.update()

        # Discriminators Update
        opt_da.zero_grad(set_to_none=True)
        opt_db.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            # Discriminator A
            loss_real_A = self.criterion_GAN(self.D_A(real_A), True)
            loss_fake_A = self.criterion_GAN(self.D_A(fake_A.detach()), False)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

        self.scaler_D_A.scale(loss_D_A).backward()
        self.scaler_D_A.step(opt_da)
        self.scaler_D_A.update()

        with torch.amp.autocast(
            device_type=device_type, dtype=torch.float16, enabled=use_amp
        ):
            # Discriminator B
            loss_real_B = self.criterion_GAN(self.D_B(real_B), True)
            loss_fake_B = self.criterion_GAN(self.D_B(fake_B.detach()), False)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

        self.scaler_D_B.scale(loss_D_B).backward()
        self.scaler_D_B.step(opt_db)
        self.scaler_D_B.update()

        metrics = {
            "loss": loss_G.item(),
            "adv_loss": loss_GAN.item(),
            "cycle_loss": loss_cycle.item(),
            "id_loss": loss_identity.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
            "d_loss": (loss_D_A + loss_D_B).item(),
        }
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.last_step_metrics = metrics

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        real_A, real_B = batch
        device_type = self.device.type
        use_amp = device_type == "cuda"

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

            loss_G = (
                loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
            )

        metrics = {
            "loss": loss_G.item(),
            "adv_loss": loss_GAN.item(),
            "cycle_loss": loss_cycle.item(),
            "id_loss": loss_identity.item(),
        }
        for k, v in metrics.items():
            self.log(
                f"val_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=False
            )

        self.validation_step_outputs.append({"loss": loss_G.detach().cpu()})
        return loss_G

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def configure_optimizers(self) -> Any:
        return [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B]
