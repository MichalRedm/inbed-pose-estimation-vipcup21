import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from src.utils.pose import decode_heatmaps


class BaseTrainer(ABC):
    """
    Abstract base class for trainers, providing common infrastructure for
    distributed training, checkpointing, and evaluation.

    Checkpoint saving is based on **validation PCK** (higher = better),
    falling back to val_loss (lower = better) if PCK is unavailable.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        # Training parameters
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.start_epoch = 0  # Default, can be set during resumption
        self.save_dir = train_cfg.get("save_dir", "results/runs/default")

        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

        # Track best by PCK (primary) and loss (fallback)
        self.best_val_pck = -1.0
        self.best_val_loss = float("inf")
        self.history = []
        self.history_path = os.path.join(self.save_dir, "history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.history = json.load(f)
            except:
                pass

    @abstractmethod
    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step and return metrics."""
        pass

    @abstractmethod
    def _val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single validation step and return metrics."""
        pass

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        metrics_sum = {}
        count = 0

        pbar = None
        if self.is_main:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

        for batch in dataloader:
            if batch is None:
                continue

            step_metrics = self._train_step(batch)

            # Accumulate metrics
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count += 1

            if pbar:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
                pbar.update(1)

        if pbar:
            pbar.close()

        # Average metrics
        avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        metrics_sum = {}
        count = 0

        for batch in dataloader:
            if batch is None:
                continue
            step_metrics = self._val_step(batch)
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count += 1

        avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}

        # In DDP, we should average metrics across all processes
        if self.world_size > 1:
            for k in avg_metrics:
                tensor = torch.tensor(avg_metrics[k], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                avg_metrics[k] = tensor.item() / self.world_size

        return avg_metrics

    @torch.no_grad()
    def compute_val_pck(self, dataloader, decode_method: str = "argmax") -> float:
        """
        Compute PCK@0.5 (torso-relative, covered validation images only).
        Used as the primary criterion for saving best_model.pth.

        Args:
            dataloader:    Validation DataLoader.
            decode_method: 'argmax' or 'soft-argmax' — must match training decoder.

        Returns:
            mean_pck: float in [0, 1].
        """
        self.model.eval()
        image_size = tuple(self.config.get("dataset", {}).get("image_size", [256, 256]))

        all_preds, all_gts, all_vis = [], [], []

        for batch in dataloader:
            if batch is None:
                continue

            images = batch["image"].to(self.device)
            joints = batch["joints"]  # (B, 3, 14)

            raw_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            outputs = raw_model(images)

            if raw_model.output_type == "heatmap":
                preds = decode_heatmaps(outputs.cpu(), image_size, method=decode_method)
            else:
                preds = outputs.cpu()

            gt_xy = joints[:, :2, :].permute(0, 2, 1).numpy()  # (B, 14, 2)
            vis = (joints[:, 2, :] <= 1).numpy()  # (B, 14) visible+occluded

            all_preds.append(preds.numpy())
            all_gts.append(gt_xy)
            all_vis.append(vis)

        if not all_preds:
            return 0.0

        P = np.concatenate(all_preds)  # (N, 14, 2)
        G = np.concatenate(all_gts)  # (N, 14, 2)
        V = np.concatenate(all_vis)  # (N, 14)

        # Torso diameter: R_Shoulder (8) to L_Hip (3)
        torso = np.linalg.norm(G[:, 8, :] - G[:, 3, :], axis=-1, keepdims=True)
        torso = np.maximum(torso, 1e-6)  # (N, 1)

        dist = np.linalg.norm(P - G, axis=-1)  # (N, 14)
        correct = (dist < 0.5 * torso) * V

        mean_pck = float(correct.sum() / np.maximum(V.sum(), 1))
        return mean_pck

    def save_checkpoint(self, name: str, is_best: bool = False):
        if not self.is_main:
            return

        checkpoint = {
            "model_state_dict": self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict(),
            "config": self.config,
            "epoch": self.current_epoch,
            "best_val_pck": self.best_val_pck,
            "best_val_loss": self.best_val_loss,
        }

        # Let subclasses add their own state (optimizers, etc.)
        checkpoint.update(self._get_extra_checkpoint_data())

        # Always save as latest for resumption
        latest_path = os.path.join(self.save_dir, "checkpoints", "latest_model.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoints", "best_model.pth")
            torch.save(checkpoint, best_path)
            if self.is_main:
                print(f"[Trainer] Saved new best model to {best_path}")

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        """Override to add optimizers, schedulers, etc."""
        return {}

    def update_history(self, epoch_data: Dict[str, Any]):
        if not self.is_main:
            return
        self.history.append(epoch_data)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=4)
