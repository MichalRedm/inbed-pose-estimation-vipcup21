import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional, Tuple, cast

from src.models.layers import SoftArgmax2D
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
    confidence_threshold: float
    lambda_unlabeled: float
    cutout_prob: float
    cutout_size_ratio: float
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

        train_cfg: Dict[str, Any] = config.get("training", {})
        self.ema_alpha = float(train_cfg.get("ema_alpha", 0.999))
        self.confidence_threshold = float(train_cfg.get("confidence_threshold", 0.35))
        self.lambda_unlabeled = float(train_cfg.get("lambda_unlabeled", 1.0))
        self.cutout_prob = float(train_cfg.get("cutout_prob", 0.5))
        self.cutout_size_ratio = float(train_cfg.get("cutout_size_ratio", 0.35))

        # 1. Create EMA Teacher as a deepcopy of the student
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.validation_step_outputs = []

    def train(self, mode: bool = True) -> "SelfTrainingLightningModule":
        """Overrides train mode to ensure the EMA teacher stays in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def on_train_start(self) -> None:
        """Called at train start to initialize weights from loop53 or identical config, and freeze teacher."""
        self.teacher.eval()

        init_weights_path = self.config.get("training", {}).get("init_weights_path", None)
        if init_weights_path:
            print(f"[SelfTraining] Initializing student and teacher from checkpoint: {init_weights_path}")
            state = torch.load(init_weights_path, map_location=self.device, weights_only=False)
            state_dict = state.get("model_state_dict", state)

            # Strip prefixes if saved from DDP or PoseDecodingWrapper
            clean_state = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "").replace("model.", "")
                clean_state[name] = v

            self.model.load_state_dict(clean_state, strict=False)

            # Synchronize teacher weights completely with the student
            with torch.no_grad():
                for s_param, t_param in zip(self.model.parameters(), self.teacher.parameters()):
                    t_param.copy_(s_param)
                for s_buffer, t_buffer in zip(self.model.buffers(), self.teacher.buffers()):
                    t_buffer.copy_(s_buffer)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x, **kwargs))

    def _get_current_sigma(self, epoch: int) -> float:
        num_epochs: int = int(self.config.get("training", {}).get("epochs", 30))
        sigma_start = float(self.config.get("training", {}).get("sigma_start", 2.0))
        sigma_end = float(self.config.get("training", {}).get("sigma_end", 2.0))
        if num_epochs <= 1:
            return sigma_start
        progress = min(epoch / (num_epochs * 0.7), 1.0)
        return sigma_start + (sigma_end - sigma_start) * progress

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        if not isinstance(batch, dict):
            return None
        batch_labeled = batch.get("labeled")
        batch_unlabeled = batch.get("unlabeled")
        if batch_labeled is None or batch_unlabeled is None:
            return None


        # -------------------------------------------------------------
        # Step 1: Labeled Loss (Supervised target)
        # -------------------------------------------------------------
        img_labeled = batch_labeled["image"]
        joints_labeled = batch_labeled["joints"]  # (B, 3, 14)
        sigma = self._get_current_sigma(self.current_epoch)

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

        # Convert peak indices to coords in 64x64 heatmap space
        H_out, W_out = 64, 64
        pred_x = (max_idx % W_out).float()
        pred_y = (max_idx // W_out).float()

        # Scale back to 256x256 image coordinate system
        scale_x = W / W_out
        scale_y = H / H_out
        pred_x = pred_x * scale_x
        pred_y = pred_y * scale_y

        # Build visibility mask using teacher confidence threshold.
        # joints with conf >= threshold are labeled visible (1), else masked (2)
        pseudo_vis = torch.where(
            conf >= self.confidence_threshold,
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
        # mask shape: (B, 14, 1, 1)
        valid_mask = (pseudo_vis <= 1).view(B, 14, 1, 1).float()
        squared_errors = (outputs_student - pseudo_targets) ** 2
        loss_unlabeled = torch.sum(squared_errors * valid_mask) / (
            torch.sum(valid_mask) * H_out * W_out + 1e-8
        )

        # -------------------------------------------------------------
        # Step 3: Combine and Balance Losses
        # -------------------------------------------------------------
        loss_total = loss_labeled + self.lambda_unlabeled * loss_unlabeled

        # Logging metrics
        metrics = {
            "loss": loss_total.item(),
            "loss_labeled": loss_labeled.item(),
            "loss_unlabeled": loss_unlabeled.item(),
            "mean_teacher_conf": conf.mean().item(),
            "conf_ratio": (pseudo_vis <= 1).float().mean().item(),
            "sigma": sigma,
        }

        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return loss_total

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Applies Exponential Moving Average (EMA) update to the Teacher weights."""
        with torch.no_grad():
            # Update learnable parameters
            for s_param, t_param in zip(self.model.parameters(), self.teacher.parameters()):
                t_param.data.mul_(self.ema_alpha).add_(s_param.data, alpha=1.0 - self.ema_alpha)
            # Update running batch norm buffers
            for s_buffer, t_buffer in zip(self.model.buffers(), self.teacher.buffers()):
                if torch.is_floating_point(t_buffer):
                    t_buffer.data.mul_(self.ema_alpha).add_(s_buffer.data, alpha=1.0 - self.ema_alpha)
                else:
                    t_buffer.data.copy_(s_buffer.data)

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
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
        from src.training.factory import build_optimizer

        optimizer = build_optimizer(self.model, self, self.config)
        return optimizer
