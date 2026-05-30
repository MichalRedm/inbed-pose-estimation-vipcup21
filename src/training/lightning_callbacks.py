import torch
import pytorch_lightning as pl
import json
import os
from typing import Dict, Any

from src.training.factory import build_optimizer


class DashboardTelemetryCallback(pl.Callback):
    """
    Custom PyTorch Lightning callback that synchronizes metrics and training status
    with the dashboard's TrainingManager and LocalTracker SQLite database.
    It also handles PCK computation and backward-compatible checkpoint saving.
    """

    def __init__(self, parent_trainer):
        super().__init__()
        self.parent = parent_trainer
        self.epoch_train_metrics = {}
        self.step_count = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.step_count = 0
        self.epoch_train_metrics = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch is None:
            return

        self.step_count += 1
        
        # Get metrics from PoseLightningModule
        metrics = getattr(pl_module, "last_step_metrics", None)
        if not metrics:
            return

        # Accumulate metrics for epoch averaging
        for k, v in metrics.items():
            self.epoch_train_metrics[k] = self.epoch_train_metrics.get(k, 0.0) + v

        # Only main process streams progress to dashboard over PTY
        if self.parent.is_main:
            # Calculate batch progress
            total_batches = len(trainer.train_dataloader) if trainer.train_dataloader else 1
            progress = self.step_count / max(total_batches, 1)

            stream_payload = {
                "epoch": trainer.current_epoch + 1,
                "progress": progress,
            }
            stream_payload.update(metrics)
            
            # Print [METRICS] and write to stream.jsonl via base class streamer
            self.parent._stream_metric(stream_payload)

    def on_train_epoch_end(self, trainer, pl_module):
        # Average training metrics
        avg_train_metrics = {
            k: v / max(self.step_count, 1) for k, v in self.epoch_train_metrics.items()
        }

        # Save to SQLite telemetry database if on main rank
        if self.parent.is_main:
            run_name = self.parent.config.get("run_id", "unnamed_run")
            for k, v in avg_train_metrics.items():
                self.parent.tracker.log_metric(run_name, trainer.current_epoch + 1, k, v)

        # Store averaged metrics on the pl_module so validation end can fetch them
        pl_module.averaged_train_metrics = avg_train_metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation epoch has finished. First, average val metrics.
        # Check if we have standard validation dataloader
        val_dataloader = None
        if trainer.val_dataloaders:
            val_dataloader = trainer.val_dataloaders
            # PyTorch Lightning handles lists or single loaders
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]

        if val_dataloader is None:
            return

        # Fetch averaged training metrics
        train_metrics = getattr(pl_module, "averaged_train_metrics", {})
        
        # Compute validation loss
        val_metrics = {}
        for k, v in trainer.callback_metrics.items():
            if k.startswith("val_"):
                # Clean key
                clean_k = k[4:]
                val_metrics[clean_k] = v.item()

        # Compute PCK@0.2 using the parent trainer's implementation
        # Set self.parent's current_epoch to match the lightning trainer's epoch
        self.parent.current_epoch = trainer.current_epoch
        
        decode_method = self.parent.config.get("training", {}).get("decode_method", "argmax")
        val_pck = self.parent.compute_val_pck(val_dataloader, decode_method=decode_method)

        # Only on main process do we log, stream summaries, and save checkpoints
        if self.parent.is_main:
            # 1. Stream comprehensive final epoch summary payload (so dashboard updates)
            summary_payload = {
                "epoch": trainer.current_epoch + 1,
                "progress": 1.0,
                "is_summary": True,
            }
            summary_payload.update(train_metrics)
            summary_payload.update({f"val_{k}": v for k, v in val_metrics.items()})
            summary_payload["val_pck"] = val_pck
            self.parent._stream_metric(summary_payload)

            # 2. Determine if it is the best model checkpoint
            val_loss = val_metrics.get("loss", float("inf"))
            is_best = val_pck > self.parent.best_val_pck
            
            if is_best:
                self.parent.best_val_pck = val_pck
            
            if val_loss < self.parent.best_val_loss:
                self.parent.best_val_loss = val_loss

            # 3. Save atomic, backward-compatible checkpoint (.pth format)
            # This uses the raw model and exactly the old dictionary layout
            self.parent.save_checkpoint(f"epoch_{trainer.current_epoch + 1}", is_best=is_best)

            # 4. Update local history.json
            epoch_data = {
                "epoch": trainer.current_epoch + 1,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "val_pck": val_pck,
            }
            self.parent.update_history(epoch_data)


class ProgressiveUnfreezingCallback(pl.Callback):
    """
    PyTorch Lightning callback that unfreezes the backbone of the model and
    rebuilds the optimizer dynamically at a specific epoch (Phase 2 Fine-Tuning).
    """

    def on_train_epoch_start(self, trainer, pl_module):
        unfreeze_epoch = pl_module.unfreeze_epoch
        if unfreeze_epoch is not None and trainer.current_epoch == unfreeze_epoch:
            # 1. Unfreeze backbone parameters
            raw_model = pl_module.model
            if hasattr(raw_model, "unfreeze_all"):
                raw_model.unfreeze_all()

            # 2. Rebuild optimizer
            new_optimizer = build_optimizer(raw_model, pl_module, pl_module.config)
            
            # 3. Replace in Trainer
            trainer.optimizers[0] = new_optimizer

            if trainer.global_rank == 0:
                print(
                    f"[Trainer] Phase 2 active at epoch {trainer.current_epoch + 1}: "
                    f"backbone unfrozen, discriminative LR applied."
                )
