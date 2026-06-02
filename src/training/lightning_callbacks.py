import pytorch_lightning as pl
from typing import Dict, Any, Optional, cast
from src.training.factory import build_optimizer


class DashboardTelemetryCallback(pl.Callback):
    """
    Custom PyTorch Lightning callback that synchronizes metrics and training status
    with the dashboard's TrainingManager and LocalTracker SQLite database.
    It also handles PCK computation and backward-compatible checkpoint saving.
    """

    parent: Any  # BaseTrainer
    epoch_train_metrics: Dict[str, float]
    step_count: int

    def __init__(self, parent_trainer: Any) -> None:
        super().__init__()
        self.parent = parent_trainer
        self.epoch_train_metrics = {}
        self.step_count = 0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.step_count = 0
        self.epoch_train_metrics = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        try:
            if batch is None:
                return

            self.step_count += 1

            # Get metrics from PoseLightningModule
            metrics: Optional[Dict[str, float]] = getattr(
                pl_module, "last_step_metrics", None
            )
            if not metrics:
                return

            # Accumulate metrics for epoch averaging
            for k, v in metrics.items():
                self.epoch_train_metrics[k] = self.epoch_train_metrics.get(k, 0.0) + v

            # Only main process streams progress to dashboard over PTY
            if self.parent.is_main:
                # Calculate batch progress
                try:
                    num_batches = trainer.num_training_batches
                    if num_batches <= 0 or num_batches == float("inf"):
                        total_batches = float(
                            len(trainer.train_dataloader)
                            if (
                                hasattr(trainer, "train_dataloader")
                                and trainer.train_dataloader
                            )
                            else 1
                        )
                    else:
                        total_batches = float(num_batches)
                except Exception:
                    total_batches = 1.0

                progress = self.step_count / max(total_batches, 1.0)

                stream_payload: Dict[str, Any] = {
                    "epoch": self.parent.start_epoch + trainer.current_epoch + 1,
                    "progress": progress,
                }
                stream_payload.update(metrics)

                # Print [METRICS] and write to stream.jsonl via base class streamer
                self.parent._stream_metric(stream_payload)
        except Exception as e:
            if self.parent.is_main:
                print(f"[Callback Error] Error in on_train_batch_end: {e}")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        try:
            # Average training metrics
            avg_train_metrics = {
                k: v / max(self.step_count, 1)
                for k, v in self.epoch_train_metrics.items()
            }

            # Save to SQLite telemetry database if on main rank
            if self.parent.is_main:
                run_name: str = self.parent.config.get("run_id", "unnamed_run")
                for k, v in avg_train_metrics.items():
                    self.parent.tracker.log_metric(
                        run_name,
                        self.parent.start_epoch + trainer.current_epoch + 1,
                        k,
                        v,
                    )

            # Fetch validation metrics compiled in on_validation_epoch_end
            val_metrics: Dict[str, float] = getattr(pl_module, "last_val_metrics", {})

            # Reset last_val_metrics on pl_module to prevent carryover
            if hasattr(pl_module, "last_val_metrics"):
                delattr(pl_module, "last_val_metrics")

            # Only on main process do we log, stream summaries, and save checkpoints
            if self.parent.is_main:
                # Sync parent trainer's epoch with PL trainer
                self.parent.current_epoch = (
                    self.parent.start_epoch + trainer.current_epoch + 1
                )

                # 1. Stream comprehensive final epoch summary payload (so dashboard updates)
                summary_payload: Dict[str, Any] = {
                    "epoch": self.parent.current_epoch,
                    "progress": 1.0,
                    "is_summary": True,
                }
                summary_payload.update(avg_train_metrics)
                summary_payload.update(val_metrics)
                self.parent._stream_metric(summary_payload)

                # 2. Determine if it is the best model checkpoint
                val_pck = float(val_metrics.get("val_pck", -1.0))
                val_loss = float(val_metrics.get("val_loss", float("inf")))
                is_best = val_pck > self.parent.best_val_pck

                if is_best:
                    self.parent.best_val_pck = val_pck

                if val_loss < self.parent.best_val_loss:
                    self.parent.best_val_loss = val_loss

                # 3. Save atomic, backward-compatible checkpoint (.pth format)
                # This uses the raw model and exactly the old dictionary layout
                try:
                    self.parent.save_checkpoint(
                        f"epoch_{self.parent.start_epoch + trainer.current_epoch + 1}",
                        is_best=is_best,
                    )
                except Exception as save_err:
                    if self.parent.is_main:
                        print(f"[Callback Error] Error saving checkpoint: {save_err}")

                # 4. Update local history.json
                try:
                    epoch_data: Dict[str, Any] = {
                        "epoch": self.parent.start_epoch + trainer.current_epoch + 1,
                        **avg_train_metrics,
                        **val_metrics,
                    }
                    self.parent.update_history(epoch_data)
                except Exception as hist_err:
                    if self.parent.is_main:
                        print(
                            f"[Callback Error] Error updating history.json: {hist_err}"
                        )
        except Exception as e:
            if self.parent.is_main:
                print(f"[Callback Error] Error in on_train_epoch_end: {e}")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Ignore validation runs triggered by sanity checking
        if getattr(trainer, "sanity_checking", False):
            return

        try:
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

            # Compute validation loss
            val_metrics: Dict[str, float] = {}
            for k, v in trainer.callback_metrics.items():
                if k.startswith("val_"):
                    # Clean key
                    clean_k = k[4:]
                    val_metrics[clean_k] = float(v.item())

            # Compute PCK@0.2 using the stashed validation predictions
            pose_module = cast(Any, pl_module)
            val_pck = -1.0

            # Only standard pose estimation trainer has PCK
            is_pose_task = (
                hasattr(pose_module, "validation_step_outputs")
                and hasattr(self.parent, "compute_val_pck")
                and not self.parent.config.get("training", {}).get("cyclegan", False)
                and self.parent.config.get("training_type", "standard") != "cyclegan"
            )

            if is_pose_task:
                if (
                    pose_module.validation_step_outputs
                    and "preds" in pose_module.validation_step_outputs[0]
                ):
                    import numpy as np

                    all_preds = []
                    all_gts = []
                    all_vis = []
                    for out in pose_module.validation_step_outputs:
                        preds = out["preds"].numpy()
                        joints = out["joints"].numpy()

                        gt_xy = joints[:, :2, :]  # (B, 2, 14)
                        gt_xy = np.transpose(gt_xy, (0, 2, 1))  # (B, 14, 2)
                        vis = joints[:, 2, :] <= 1  # (B, 14) visible + occluded

                        all_preds.append(preds)
                        all_gts.append(gt_xy)
                        all_vis.append(vis)

                    P = np.concatenate(all_preds, axis=0)  # (N, 14, 2)
                    G = np.concatenate(all_gts, axis=0)  # (N, 14, 2)
                    V = np.concatenate(all_vis, axis=0)  # (N, 14)

                    # Torso diameter: R_Shoulder (8) to L_Hip (3)
                    torso = np.linalg.norm(
                        G[:, 8, :] - G[:, 3, :], axis=-1, keepdims=True
                    )
                    torso = np.maximum(torso, 1e-6)  # (N, 1)

                    dist_val = np.linalg.norm(P - G, axis=-1)  # (N, 14)
                    correct = (dist_val < 0.2 * torso) * V

                    val_pck = float(correct.sum() / np.maximum(V.sum(), 1))

                    # Clear validation step outputs to save memory
                    pose_module.validation_step_outputs = []
                elif val_dataloader is not None:
                    # Fallback to compute_val_pck (original slow method)
                    decode_method: str = self.parent.config.get("training", {}).get(
                        "decode_method", "argmax"
                    )
                    val_pck = self.parent.compute_val_pck(
                        val_dataloader, decode_method=decode_method
                    )

            # Store on pl_module to be fetched by on_train_epoch_end
            setattr(
                pl_module,
                "last_val_metrics",
                {
                    "val_pck": val_pck,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                },
            )
        except Exception as e:
            if self.parent.is_main:
                print(f"[Callback Error] Error in on_validation_epoch_end: {e}")


class ProgressiveUnfreezingCallback(pl.Callback):
    """
    PyTorch Lightning callback that unfreezes the backbone of the model and
    rebuilds the optimizer dynamically at a specific epoch (Phase 2 Fine-Tuning).
    """

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        from src.training.lightning_module import PoseLightningModule

        pose_module = cast(PoseLightningModule, pl_module)

        unfreeze_epoch: Optional[int] = pose_module.unfreeze_epoch
        if unfreeze_epoch is not None and trainer.current_epoch == unfreeze_epoch:
            # 1. Unfreeze backbone parameters
            raw_model = pose_module.model
            if hasattr(raw_model, "unfreeze_all"):
                # Use cast or ignore if unfreeze_all is not in nn.Module but in our HRNet
                getattr(raw_model, "unfreeze_all")()

            # 2. Rebuild optimizer
            new_optimizer = build_optimizer(raw_model, pose_module, pose_module.config)

            # 3. Replace in Trainer
            trainer.optimizers[0] = new_optimizer

            if trainer.global_rank == 0:
                print(
                    f"[Trainer] Phase 2 active at epoch {trainer.current_epoch + 1}: "
                    f"backbone unfrozen, discriminative LR applied."
                )
