import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.training.base_trainer import BaseTrainer
from src.data.dataset import VIPCupDataset
from src.data.augmentations import DataAugmenter


class SelfTrainingTrainer(BaseTrainer):
    """
    Orchestration trainer for Teacher-Student Self-Training (Pseudo-Labeling).
    Handles dataloader construction for unannotated target subjects (31-80) and
    manages the PyTorch Lightning execution setup.
    """

    optimizer: torch.optim.Optimizer
    criterion: nn.Module

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__(model, config, device, rank, world_size)
        if optimizer is not None:
            self.optimizer = optimizer
        self.criterion = criterion

    def _train_step(self, batch: Any) -> Dict[str, float]:
        # Not used since PyTorch Lightning handles steps under the hood,
        # but required to satisfy abstract method constraints in BaseTrainer.
        return {"loss": 0.0}

    def _val_step(self, batch: Any) -> Dict[str, float]:
        # Not used since PyTorch Lightning handles steps under the hood,
        # but required to satisfy abstract method constraints in BaseTrainer.
        return {"loss": 0.0}

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        extra_data: Dict[str, Any] = {
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        # Save teacher weights if available (found inside the PL lightning module)
        if hasattr(self, "lightning_module") and hasattr(
            self.lightning_module, "teacher"
        ):
            extra_data["teacher_state_dict"] = (
                self.lightning_module.teacher.state_dict()
            )

        # Save running teacher confidence buffer for curriculum resumption stability
        if hasattr(self, "lightning_module") and hasattr(
            self.lightning_module, "running_teacher_conf"
        ):
            extra_data["running_teacher_conf"] = (
                self.lightning_module.running_teacher_conf.item()
            )

        return extra_data

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        """
        Builds the unannotated loader, packages it with the labeled loader,
        and launches the PyTorch Lightning trainer.
        """
        from .self_training_lightning import SelfTrainingLightningModule
        from .lightning_callbacks import DashboardTelemetryCallback

        # 1. Build unannotated target dataset and loader
        dataset_cfg = self.config.get("dataset", {})
        data_root = dataset_cfg.get("dataset_root", "data/SLP")
        image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

        # Clean DataAugmenter for unlabeled target data:
        # NO cover/occlusion simulations (as target subjects already have physical blankets)
        unlabeled_aug_cfg = self.config["training"].get("augmentation", {}).copy()  # type: ignore[assignment]
        unlabeled_aug_cfg["occlusion_prob"] = 0.0
        unlabeled_aug_cfg["advanced_cover_prob"] = 0.0
        unlabeled_aug_cfg["cyclegan_prob"] = 0.0
        unlabeled_aug_cfg["cut_prob"] = 0.0

        if self.is_main:
            print(
                "[SelfTrainingTrainer] Creating clean augmenter for unannotated examples..."
            )
        unlabeled_augmenter = DataAugmenter(unlabeled_aug_cfg, dataset_root=data_root)

        # Subjects 31-80 are unannotated target domain
        unlabeled_dataset = VIPCupDataset(
            root=data_root,
            subjects=range(31, 81),
            covers=["cover1", "cover2"],
            modalities=["IR"],
            split="train",
            augmenter=unlabeled_augmenter,
            image_size=image_size,
            in_channels=int(getattr(self.model, "in_channels", 3)),
        )

        from src.data.dataset import collate_unlabeled

        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=self.config["training"].get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_unlabeled,
            num_workers=4 if os.name != "nt" else 0,
        )

        if self.is_main:
            print(f"[SelfTrainingTrainer] Labeled samples: {len(train_loader.dataset)}")
            print(f"[SelfTrainingTrainer] Unlabeled samples: {len(unlabeled_dataset)}")

        # 2. Package loaders with min_size mode so neither loader returns None
        # when the shorter one is exhausted (PL default max_size fills with None).
        from pytorch_lightning.utilities import CombinedLoader

        combined_loaders = CombinedLoader(
            {"labeled": train_loader, "unlabeled": unlabeled_loader},
            mode="min_size",
        )

        # 3. Instantiate Lightning Module
        self.lightning_module = SelfTrainingLightningModule(
            model=self.model,
            config=self.config,
            criterion=self.criterion,
            optimizer=self.optimizer,
        )

        # 4. Instantiate custom callbacks
        callbacks = [DashboardTelemetryCallback(self)]

        # 5. Configure PL Trainer options
        trainer = self._setup_pl_trainer(callbacks=callbacks)

        if self.is_main:
            print(
                "[SelfTrainingTrainer] Starting PyTorch Lightning self-training loop..."
            )
            print(
                f"[SelfTrainingTrainer] Accelerator: {trainer.accelerator}, Devices: {trainer.num_devices}, Strategy: {trainer.strategy}"
            )

        # 6. Restore state if resuming
        if self.resume_state:
            self._load_extra_checkpoint_data(self.resume_state)

        # Start training
        self._run_pl_fit(trainer, self.lightning_module, combined_loaders, val_loader)

    def _load_extra_checkpoint_data(self, state: Dict[str, Any]) -> None:
        if "optimizer_state_dict" in state:
            if self.is_main:
                print("[SelfTrainingTrainer] Restoring optimizer state.")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if "teacher_state_dict" in state and hasattr(self, "lightning_module"):
            if self.is_main:
                print("[SelfTrainingTrainer] Restoring EMA teacher weights.")
            self.lightning_module.teacher.load_state_dict(state["teacher_state_dict"])

        if "running_teacher_conf" in state and hasattr(self, "lightning_module"):
            if self.is_main:
                print(
                    f"[SelfTrainingTrainer] Restoring running_teacher_conf: {state['running_teacher_conf']:.4f}"
                )
            self.lightning_module.running_teacher_conf.copy_(
                torch.tensor(state["running_teacher_conf"], dtype=torch.float32)
            )
