import copy
import math
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, cast
from src.training.standard_trainer import generate_pytorch_heatmaps
from src.utils import decode_heatmaps


def apply_tensor_cutout(
    img_tensor: torch.Tensor,
    scale_ratio: float = 0.35,
    probability: float = 0.5,
) -> torch.Tensor:
    """
    Applies random patch erasing (cutout) to a batch of image tensors on the GPU.
    img_tensor shape: (B, C, H, W)
    """
    if random.random() > probability:
        return img_tensor

    B, C, H, W = img_tensor.shape
    out = img_tensor.clone()
    for i in range(B):
        # Determine cutout size
        cut_h = int(H * scale_ratio)
        cut_w = int(W * scale_ratio)

        # Random top-left corner
        cy = torch.randint(0, H - cut_h, (1,)).item()
        cx = torch.randint(0, W - cut_w, (1,)).item()

        # Fill with zero (emulates background / lack of information)
        out[i, :, cy : cy + cut_h, cx : cx + cut_w] = 0.0
    return out


class SelfTrainingLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module wrapping the Teacher-Student Self-Training (Pseudo-Labeling) algorithm.
    Coordinates labeled task loss and unlabeled joint-gated consistency regularization with EMA updates.
    """

    model: nn.Module
    teacher: nn.Module
    config: Dict[str, Any]
    criterion: nn.Module
    ema_alpha: float
    ema_alpha_start: float
    ema_alpha_end: float
    conf_threshold_start: float
    conf_threshold_end: float
    lambda_unlabeled: float
    cutout_prob: float
    cutout_size_ratio: float
    running_teacher_conf: torch.Tensor
    last_step_metrics: Dict[str, float]
    validation_step_outputs: List[Dict[str, torch.Tensor]]

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()
        self.external_optimizer = optimizer

        train_cfg: Dict[str, Any] = config.get("training", {})
        self.ema_alpha = float(train_cfg.get("ema_alpha", 0.999))
        self.ema_alpha_start = float(train_cfg.get("ema_alpha_start", 0.99))
        self.ema_alpha_end = float(train_cfg.get("ema_alpha_end", self.ema_alpha))
        static_thresh = float(train_cfg.get("confidence_threshold", 0.35))
        self.conf_threshold_start = float(
            train_cfg.get("conf_threshold_start", static_thresh)
        )
        self.conf_threshold_end = float(
            train_cfg.get("conf_threshold_end", static_thresh)
        )
        self.lambda_unlabeled = float(train_cfg.get("lambda_unlabeled", 1.0))
        self.cutout_prob = float(train_cfg.get("cutout_prob", 0.5))
        self.cutout_size_ratio = float(train_cfg.get("cutout_size_ratio", 0.35))

        # Register running teacher confidence buffer for adaptive thresholding
        self.register_buffer(
            "running_teacher_conf", torch.tensor(0.65, dtype=torch.float32)
        )

        # 1. Create EMA Teacher as a deepcopy of the student
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.last_step_metrics = {}
        self.validation_step_outputs = []

    def train(self, mode: bool = True) -> "SelfTrainingLightningModule":
        """Overrides train mode to ensure the EMA teacher stays in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def on_train_start(self) -> None:
        """Called at train start to initialize weights from loop53 or identical config, and freeze teacher."""
        self.teacher.eval()

        # Only initialize from loop53 if we are starting a fresh run.
        # If we are resuming, weights have already been restored from the latest checkpoint.
        if self.current_epoch > 0:
            return

        init_weights_path = self.config.get("training", {}).get(
            "init_weights_path", None
        )
        if init_weights_path:
            print(
                f"[SelfTraining] Initializing student and teacher from checkpoint: {init_weights_path}"
            )
            state = torch.load(
                init_weights_path, map_location=self.device, weights_only=False
            )
            state_dict = state.get("model_state_dict", state)

            # Strip prefixes if saved from DDP or PoseDecodingWrapper
            clean_state = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "").replace("model.", "")
                clean_state[name] = v

            self.model.load_state_dict(clean_state, strict=False)

            # Synchronize teacher weights completely with the student
            with torch.no_grad():
                for s_param, t_param in zip(
                    self.model.parameters(), self.teacher.parameters()
                ):
                    t_param.copy_(s_param)
                for s_buffer, t_buffer in zip(
                    self.model.buffers(), self.teacher.buffers()
                ):
                    t_buffer.copy_(s_buffer)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x, **kwargs))

    @property
    def global_epoch(self) -> int:
        return getattr(self, "global_start_epoch", 0) + self.current_epoch

    def _get_current_confidence_threshold(self, epoch: int) -> float:
        train_cfg = self.config.get("training", {})
        if "teacher_conf_min" in train_cfg or train_cfg.get(
            "dynamic_confidence_threshold", False
        ):
            t_min = float(train_cfg.get("teacher_conf_min", 0.65))
            t_max = float(train_cfg.get("teacher_conf_max", 0.85))
            norm_conf = (self.running_teacher_conf.item() - t_min) / (t_max - t_min)
            norm_conf = max(0.0, min(1.0, norm_conf))

            # Linearly interpolate between start and end thresholds based on teacher confidence
            return self.conf_threshold_start + norm_conf * (
                self.conf_threshold_end - self.conf_threshold_start
            )

        num_epochs: int = int(train_cfg.get("epochs", 30))
        if num_epochs <= 1:
            return self.conf_threshold_start
        progress = min(epoch / (num_epochs - 1), 1.0)
        # Cosine decay from conf_threshold_start to conf_threshold_end
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.conf_threshold_end
            + (self.conf_threshold_start - self.conf_threshold_end) * cosine_decay
        )

    def _get_current_ema_alpha(self, epoch: int) -> float:
        num_epochs: int = int(self.config.get("training", {}).get("epochs", 30))
        if num_epochs <= 1:
            return self.ema_alpha_end
        progress = min(epoch / (num_epochs - 1), 1.0)
        # Cosine scheduling from ema_alpha_start to ema_alpha_end
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.ema_alpha_end
            + (self.ema_alpha_start - self.ema_alpha_end) * cosine_val
        )

    def _get_joint_specific_thresholds(self, base_threshold: float) -> torch.Tensor:
        # 14 joints mapping:
        # 0: R_Ankle (Extremity, 0.70)
        # 1: R_Knee (Limb, 0.85)
        # 2: R_Hip (Core, 1.00)
        # 3: L_Hip (Core, 1.00)
        # 4: L_Knee (Limb, 0.85)
        # 5: L_Ankle (Extremity, 0.70)
        # 6: R_Wrist (Extremity, 0.70)
        # 7: R_Elbow (Limb, 0.85)
        # 8: R_Shoulder (Core, 1.00)
        # 9: L_Shoulder (Core, 1.00)
        # 10: L_Elbow (Limb, 0.85)
        # 11: L_Wrist (Extremity, 0.70)
        # 12: Thorax (Core, 1.00)
        # 13: Head (Core, 1.00)
        discounts = torch.tensor(
            [
                0.70,
                0.85,
                1.00,
                1.00,
                0.85,
                0.70,
                0.70,
                0.85,
                1.00,
                1.00,
                0.85,
                0.70,
                1.00,
                1.00,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return base_threshold * discounts

    def _get_current_sigma(self, epoch: int) -> float:
        num_epochs: int = int(self.config.get("training", {}).get("epochs", 30))
        sigma_start = float(self.config.get("training", {}).get("sigma_start", 2.0))
        sigma_end = float(self.config.get("training", {}).get("sigma_end", 2.0))
        if num_epochs <= 1:
            return sigma_start
        progress = min(epoch / (num_epochs * 0.7), 1.0)
        return sigma_start + (sigma_end - sigma_start) * progress

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        if batch is None:
            return None

        # Handle both dict and non-dict batch (DDP fallback)
        if not isinstance(batch, dict):
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        batch_labeled = batch.get("labeled")
        batch_unlabeled = batch.get("unlabeled")

        if batch_labeled is None or batch_unlabeled is None:
            # If CombinedLoader failed to provide both, we can't do self-training
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # -------------------------------------------------------------
        # Step 1: Labeled Loss (Supervised target)
        # -------------------------------------------------------------
        img_labeled = batch_labeled["image"]
        joints_labeled = batch_labeled["joints"]  # (B, 3, 14)

        # Use global_epoch for accurate curriculum progress
        sigma = self._get_current_sigma(self.global_epoch)
        current_conf_threshold = self._get_current_confidence_threshold(
            self.global_epoch
        )
        joint_thresholds = self._get_joint_specific_thresholds(current_conf_threshold)
        current_ema_alpha = self._get_current_ema_alpha(self.global_epoch)

        # Generate target heatmaps on GPU
        targets_labeled = generate_pytorch_heatmaps(
            joints=joints_labeled,
            heatmap_size=(64, 64),
            image_size=(256, 256),
            sigma=sigma,
        )

        outputs_labeled = self.model(img_labeled)
        loss_labeled = self.criterion(outputs_labeled, targets_labeled)

        # -------------------------------------------------------------
        # Step 2: Unlabeled Loss (Teacher-Student Pseudo-Labeling)
        # -------------------------------------------------------------
        img_unlabeled_weak = batch_unlabeled["image"]
        B, C, H, W = img_unlabeled_weak.shape

        # 2.1 Teacher Predicts Pseudo-labels on Weak View
        with torch.no_grad():
            self.teacher.eval()
            pred_heatmaps_teacher = self.teacher(img_unlabeled_weak)  # (B, 14, 64, 64)

        # 2.2 Decode peak coordinates and confidence thresholds on GPU
        flat_heatmaps = pred_heatmaps_teacher.view(B, 14, -1)
        conf, max_idx = flat_heatmaps.max(dim=-1)  # conf: (B, 14), max_idx: (B, 14)

        # Update running teacher confidence EMA
        batch_mean_conf = conf.mean()
        self.running_teacher_conf = (
            0.99 * self.running_teacher_conf + 0.01 * batch_mean_conf.detach()
        )

        # Convert peak indices to coords in 64x64 heatmap space
        H_out, W_out = 64, 64
        pred_x = (max_idx % W_out).float()
        pred_y = (max_idx // W_out).float()

        # Scale back to 256x256 image coordinate system
        scale_x = W / W_out
        scale_y = H / H_out
        pred_x = pred_x * scale_x
        pred_y = pred_y * scale_y

        # Build visibility mask using joint-specific thresholds.
        # joints with conf >= threshold are labeled visible (1), else masked (2)
        pseudo_vis = torch.where(
            conf >= joint_thresholds.unsqueeze(0),
            torch.ones_like(conf),
            torch.ones_like(conf) * 2,
        )

        # Stack into (B, 3, 14) format
        pseudo_joints = torch.stack([pred_x, pred_y, pseudo_vis], dim=1)

        # 2.3 Render pseudo-target Gaussian heatmaps dynamically
        pseudo_targets = generate_pytorch_heatmaps(
            joints=pseudo_joints,
            heatmap_size=(64, 64),
            image_size=(256, 256),
            sigma=sigma,
        )

        # 2.4 Strong Augmentation for Student on GPU (cutout + noise)
        img_unlabeled_strong = apply_tensor_cutout(
            img_unlabeled_weak,
            scale_ratio=self.cutout_size_ratio,
            probability=self.cutout_prob,
        )

        # Student Forward on Strong View
        outputs_student = self.model(img_unlabeled_strong)

        # 2.5 Dynamic Masked Loss: Only compute regression for confident keypoints
        # mask shape: (B, 14, 1, 1), conf shape: (B, 14) -> (B, 14, 1, 1)
        valid_mask = (pseudo_vis <= 1).view(B, 14, 1, 1).float()
        soft_weight = conf.view(B, 14, 1, 1)

        squared_errors = (outputs_student - pseudo_targets) ** 2
        loss_unlabeled = torch.sum(squared_errors * valid_mask * soft_weight) / (
            torch.sum(valid_mask) * H_out * W_out + 1e-8
        )

        # -------------------------------------------------------------
        # Step 3: Combine and Balance Losses
        # -------------------------------------------------------------
        train_cfg = self.config.get("training", {})
        t_min = float(train_cfg.get("teacher_conf_min", 0.65))
        t_max = float(train_cfg.get("teacher_conf_max", 0.85))
        norm_conf = (self.running_teacher_conf.item() - t_min) / (t_max - t_min)
        norm_conf = max(0.0, min(1.0, norm_conf))

        if train_cfg.get("dynamic_lambda_unlabeled", False):
            lambda_min = float(train_cfg.get("lambda_unlabeled_min", 0.2))
            lambda_max = float(train_cfg.get("lambda_unlabeled_max", 1.5))
            current_lambda = lambda_min + norm_conf * (lambda_max - lambda_min)
        else:
            current_lambda = self.lambda_unlabeled

        loss_total = loss_labeled + current_lambda * loss_unlabeled

        # Logging metrics
        metrics = {
            "loss": loss_total.item(),
            "loss_labeled": loss_labeled.item(),
            "loss_unlabeled": loss_unlabeled.item(),
            "mean_teacher_conf": conf.mean().item(),
            "running_teacher_conf": self.running_teacher_conf.item(),
            "norm_teacher_conf": norm_conf,
            "lambda_unlabeled": current_lambda,
            "conf_ratio": (pseudo_vis <= 1).float().mean().item(),
            "sigma": sigma,
            "conf_threshold": current_conf_threshold,
            "ema_alpha": current_ema_alpha,
        }

        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        self.last_step_metrics = metrics
        return cast(Optional[torch.Tensor], loss_total)

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Applies Exponential Moving Average (EMA) update to the Teacher weights."""
        current_ema_alpha = self._get_current_ema_alpha(self.global_epoch)
        with torch.no_grad():
            # Update learnable parameters
            for s_param, t_param in zip(
                self.model.parameters(), self.teacher.parameters()
            ):
                t_param.data.mul_(current_ema_alpha).add_(
                    s_param.data, alpha=1.0 - current_ema_alpha
                )
            # Update running batch norm buffers
            for s_buffer, t_buffer in zip(self.model.buffers(), self.teacher.buffers()):
                if torch.is_floating_point(t_buffer):
                    t_buffer.data.mul_(current_ema_alpha).add_(
                        s_buffer.data, alpha=1.0 - current_ema_alpha
                    )
                else:
                    t_buffer.data.copy_(s_buffer.data)

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Optional[torch.Tensor]:
        if batch is None:
            return None

        images = batch["image"]
        joints = batch["joints"]

        sigma = self._get_current_sigma(self.current_epoch)
        targets = generate_pytorch_heatmaps(
            joints=joints,
            heatmap_size=(64, 64),
            image_size=(256, 256),
            sigma=sigma,
        )

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

        # Decode validation predictions for strict epoch-end PCK callback
        method = self.config.get("training", {}).get("decode_method", "argmax")
        temp = float(self.config.get("training", {}).get("decode_temperature", 10.0))
        preds = decode_heatmaps(outputs, (256, 256), method=method, temperature=temp)

        self.validation_step_outputs.append(
            {"preds": preds.detach().cpu(), "joints": joints.detach().cpu()}
        )

        return cast(torch.Tensor, loss)

    def configure_optimizers(self) -> Any:
        if self.external_optimizer is not None:
            return self.external_optimizer

        from src.training.factory import build_optimizer

        optimizer = build_optimizer(self.model, self, self.config)
        return optimizer
