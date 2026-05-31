"""
Abstract base trainer providing foundational training and evaluation infrastructure.
Handles checkpointing, distributed coordination, and real-time metric streaming.
"""

import os
import json
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, cast
import numpy as np

from src.utils.pose import decode_heatmaps
from src.utils.telemetry import LocalTracker, JSONLStream


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.

    Provides:
      - Distributed training coordination.
      - Atomic checkpoint saving with verification.
      - Real-time metric streaming for dashboards.
      - PCK-based model selection.
    """

    model: torch.nn.Module
    config: Dict[str, Any]
    device: torch.device
    rank: int
    world_size: int
    is_main: bool
    epochs: int
    start_epoch: int
    current_epoch: int
    save_dir: Optional[str]
    best_val_pck: float
    best_val_loss: float
    history: List[Dict[str, Any]]
    history_path: Optional[str]
    stream_path: Optional[str]
    streamer: Optional[JSONLStream]
    tracker: LocalTracker

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initializes the BaseTrainer.

        Args:
            model: The PyTorch model to train.
            config: Full project configuration.
            device: Torch device to use.
            rank: Process rank for distributed training.
            world_size: Total number of processes.
        """
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        # Training parameters
        train_cfg: Dict[str, Any] = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 30))
        self.start_epoch = 0  # Default, can be set during resumption
        self.current_epoch = 0
        self.save_dir = train_cfg.get("save_dir")

        if self.is_main and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

        # Track best by PCK (primary) and loss (fallback)
        self.best_val_pck = -1.0
        self.best_val_loss = float("inf")
        self.history = []
        self.history_path = (
            os.path.join(self.save_dir, "history.json") if self.save_dir else None
        )
        if self.history_path and os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.history = json.load(f)
            except Exception:
                pass

        # Dedicated JSON stream for real-time dashboard updates
        self.stream_path = (
            os.path.join(self.save_dir, "stream.jsonl") if self.save_dir else None
        )
        self.streamer = JSONLStream(self.stream_path) if self.stream_path else None

        # Local SQLite Tracker
        self.tracker = LocalTracker()
        if self.is_main:
            run_name: str = config.get("run_id", "unnamed_run")
            self.tracker.init_run(run_name, run_name, config)

    def _stream_metric(self, data: Dict[str, Any]) -> None:
        """
        Append a JSON line to the stream file for real-time telemetry.

        Args:
            data: Metric dictionary to stream.
        """
        if not self.is_main or not self.streamer:
            return

        # Rate-limit batch-level metrics to reduce disk I/O and CPU overhead
        is_summary = bool(data.get("is_summary", False))
        progress = float(data.get("progress", 0.0))
        
        if not is_summary:
            if not hasattr(self, "_step_stream_counter"):
                self._step_stream_counter = 0
            self._step_stream_counter += 1
            
            # Send on the first batch of the session, every 10 batches, and at epoch end
            if self._step_stream_counter % 10 != 1 and progress < 1.0:
                return

        # Inject display metadata periodically (start of epoch OR every 10% progress)
        is_start = progress <= 0.01

        if not hasattr(self, "_last_metadata_progress"):
            self._last_metadata_progress = -1.0

        # Send if it's the start, if we haven't sent it yet, or every 10% progress increment
        if (
            is_start
            or not hasattr(self, "_metadata_sent")
            or (progress - self._last_metadata_progress) >= 0.10
        ):
            data["display_metadata"] = self.get_display_metadata()
            self._metadata_sent = True
            self._last_metadata_progress = progress

        self.streamer.emit(data)

    def get_display_metadata(self) -> Dict[str, Any]:
        """Return hints for the frontend dashboard on how to display metrics."""
        from src.utils.config_manager import get_display_metadata_for_config

        return get_display_metadata_for_config(self.config)

    @abstractmethod
    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Perform a single training step and return metrics."""
        pass

    @abstractmethod
    def _val_step(self, batch: Any) -> Dict[str, float]:
        """Perform a single validation step and return metrics."""
        pass

    def train_epoch(self, dataloader: Any, epoch: int) -> Dict[str, float]:
        """
        Trains the model for one epoch.

        Args:
            dataloader: Training dataloader.
            epoch: Current epoch index.

        Returns:
            Dictionary of average training metrics.
        """
        self.model.train()
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        metrics_sum: Dict[str, float] = {}
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

                # Stream JSON metrics to sidecar file
                stream_payload: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "progress": count / max(len(dataloader), 1),
                }
                stream_payload.update(step_metrics)
                self._stream_metric(stream_payload)

        if pbar:
            pbar.close()

        # Average metrics
        avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}

        if self.is_main:
            # Persistent SQLite logging
            run_name: str = self.config.get("run_id", "unnamed_run")
            for k, v in avg_metrics.items():
                self.tracker.log_metric(run_name, epoch + 1, k, v)

        # Stream final epoch summary
        summary_payload: Dict[str, Any] = {
            "epoch": epoch + 1,
            "progress": 1.0,
            "is_summary": True,
        }
        summary_payload.update(avg_metrics)
        self._stream_metric(summary_payload)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        self.model.eval()
        metrics_sum: Dict[str, float] = {}
        count = 0

        pbar = None
        if self.is_main:
            pbar = tqdm(dataloader, desc="Validation", leave=False)

        for batch in dataloader:
            if batch is None:
                continue
            step_metrics = self._val_step(batch)
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count += 1

            if pbar:
                pbar.update(1)
                # Optional: stream validation progress too
                self._stream_metric(
                    {"phase": "val", "progress": count / max(len(dataloader), 1)}
                )

        if pbar:
            pbar.close()

        avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}

        # In DDP, we should average metrics across all processes
        if self.world_size > 1:
            for k in avg_metrics:
                val_tensor = torch.tensor(avg_metrics[k], device=self.device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                avg_metrics[k] = val_tensor.item() / self.world_size

        return avg_metrics

    @torch.no_grad()
    def compute_val_pck(
        self, dataloader: Any, decode_method: Optional[str] = None
    ) -> float:
        """
        Compute PCK@0.2 (torso-relative).
        Evaluates on visible and occluded joints (vis <= 1).

        Args:
            dataloader: Validation dataloader.
            decode_method: Optional override for heatmap decoding method.

        Returns:
            Mean PCK@0.2 across the dataset.
        """
        self.model.eval()
        image_size: Tuple[int, int] = tuple(
            self.config.get("dataset", {}).get("image_size", [256, 256])
        )

        all_preds, all_gts, all_vis = [], [], []

        pbar = None
        if self.is_main:
            pbar = tqdm(dataloader, desc="PCK Eval", leave=False)

        for batch in dataloader:
            if batch is None:
                continue

            images = batch["image"].to(self.device)
            joints = batch["joints"]  # (B, 3, 14)

            if pbar:
                pbar.update(1)

            raw_model = cast(
                nn.Module,
                self.model.module if hasattr(self.model, "module") else self.model,
            )
            outputs = raw_model(images)

            if getattr(raw_model, "output_type", "heatmap") == "heatmap":
                method = decode_method or self.config.get("training", {}).get(
                    "decode_method", "argmax"
                )
                temp = float(
                    self.config.get("training", {}).get("decode_temperature", 10.0)
                )
                preds = decode_heatmaps(
                    outputs, image_size, method=method, temperature=temp
                ).cpu()
            else:
                preds = outputs.cpu()

            gt_xy = joints[:, :2, :].permute(0, 2, 1).numpy()  # (B, 14, 2)
            vis = (joints[:, 2, :] <= 1).numpy()  # (B, 14) visible+occluded

            all_preds.append(preds.numpy())
            all_gts.append(gt_xy)
            all_vis.append(vis)

        if pbar:
            pbar.close()

        if not all_preds:
            return 0.0

        P = np.concatenate(all_preds)  # (N, 14, 2)
        G = np.concatenate(all_gts)  # (N, 14, 2)
        V = np.concatenate(all_vis)  # (N, 14)

        # Torso diameter: R_Shoulder (8) to L_Hip (3)
        torso = np.linalg.norm(G[:, 8, :] - G[:, 3, :], axis=-1, keepdims=True)
        torso = np.maximum(torso, 1e-6)  # (N, 1)

        dist_val = np.linalg.norm(P - G, axis=-1)  # (N, 14)
        correct = (dist_val < 0.2 * torso) * V

        mean_pck = float(correct.sum() / np.maximum(V.sum(), 1))
        return mean_pck

    def save_checkpoint(self, name: str, is_best: bool = False) -> None:
        """
        Saves a model checkpoint with configuration and metrics.
        Uses atomic saving with verification to prevent corrupted files.

        Args:
            name: Checkpoint name (used for filename).
            is_best: If True, also copies the checkpoint to 'best_model.pth'.
        """
        if not self.is_main or not self.save_dir:
            return

        model_to_save = cast(
            nn.Module,
            self.model.module if hasattr(self.model, "module") else self.model,
        )

        checkpoint: Dict[str, Any] = {
            "model_state_dict": model_to_save.state_dict(),
            "config": self.config,
            "epoch": self.current_epoch,
            "best_val_pck": self.best_val_pck,
            "best_val_loss": self.best_val_loss,
            "decoding_config": {
                "method": self.config.get("training", {}).get(
                    "decode_method", "argmax"
                ),
                "temperature": self.config.get("training", {}).get(
                    "decode_temperature", 10.0
                ),
                "image_size": self.config.get("dataset", {}).get(
                    "image_size", [256, 256]
                ),
            },
        }

        # Let subclasses add their own state (optimizers, etc.)
        checkpoint.update(self._get_extra_checkpoint_data())

        def _atomic_torch_save(obj: Dict[str, Any], target_path: str) -> bool:
            tmp_path = str(target_path) + ".tmp"

            # 1. Save to temporary file
            torch.save(obj, tmp_path)

            # 2. VERIFY the saved file before committing
            try:
                # We only need to check if the zip archive is valid
                # map_location='cpu' and weights_only=True for speed
                torch.load(tmp_path, map_location="cpu", weights_only=True)
            except Exception as e:
                print(
                    f"[Trainer] CRITICAL: Verification of saved checkpoint failed: {e}"
                )
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return False  # Indicate failure

            # 3. Commit (with retry for Windows locking)
            import time

            for i in range(5):
                try:
                    if os.path.exists(target_path):
                        os.replace(tmp_path, target_path)
                    else:
                        os.rename(tmp_path, target_path)
                    return True
                except PermissionError:
                    if i == 4:
                        print(
                            f"[Trainer] ERROR: Could not commit checkpoint to {target_path} (Locked)"
                        )
                        return False
                    time.sleep(0.5)
            return False

        # Always save as latest for resumption
        latest_path = os.path.join(self.save_dir, "checkpoints", "latest_model.pth")
        if _atomic_torch_save(checkpoint, latest_path):
            if is_best:
                best_path = os.path.join(self.save_dir, "checkpoints", "best_model.pth")
                try:
                    import shutil

                    shutil.copy2(latest_path, best_path)
                    if self.is_main:
                        print(
                            f"[Trainer] Verified and saved new best model to {best_path}"
                        )
                except Exception as e:
                    if self.is_main:
                        print(f"[Trainer] Warning: could not copy best model: {e}")

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        """Override to add optimizers, schedulers, etc."""
        return {}

    def update_history(self, epoch_data: Dict[str, Any]) -> None:
        """
        Appends epoch metrics to the history and saves it to a JSON file.
        Uses atomic saving to prevent history loss.

        Args:
            epoch_data: Metrics for the current epoch.
        """
        if not self.is_main or not self.history_path:
            return
        self.history.append(epoch_data)

        tmp_history = str(self.history_path) + ".tmp"
        with open(tmp_history, "w") as f:
            json.dump(self.history, f, indent=4)

        import os

        os.replace(tmp_history, self.history_path)
