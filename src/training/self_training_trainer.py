import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.training.base_trainer import BaseTrainer
from src.data.dataset import VIPCupDataset, collate_skip_none
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
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def fit(self, train_loader: Any, val_loader: Any = None) -> None:
        """
        Builds the unannotated loader, packages it with the labeled loader,
        and launches the PyTorch Lightning trainer.
        """
        from .self_training_lightning import SelfTrainingLightningModule
        from .lightning_callbacks import DashboardTelemetryCallback
        import pytorch_lightning as pl

        # 1. Build unannotated target dataset and loader
        dataset_cfg = self.config.get("dataset", {})
        data_root = dataset_cfg.get("dataset_root", "data/SLP")
        image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

        # Clean DataAugmenter for unlabeled target data:
        # NO cover/occlusion simulations (as target subjects already have physical blankets)
        unlabeled_aug_cfg = self.config["training"].get("augmentation", {}).copy()
        unlabeled_aug_cfg["occlusion_prob"] = 0.0
        unlabeled_aug_cfg["advanced_cover_prob"] = 0.0
        unlabeled_aug_cfg["cyclegan_prob"] = 0.0
        unlabeled_aug_cfg["cut_prob"] = 0.0

        if self.is_main:
            print("[SelfTrainingTrainer] Creating clean augmenter for unannotated examples...")
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
            in_channels=self.model.in_channels if hasattr(self.model, "in_channels") else 3,
        )

        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=self.config["training"].get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_skip_none,
            num_workers=4 if os.name != "nt" else 0,
        )

        if self.is_main:
            print(f"[SelfTrainingTrainer] Labeled samples: {len(train_loader.dataset)}")
            print(f"[SelfTrainingTrainer] Unlabeled samples: {len(unlabeled_dataset)}")

        # 2. Package loaders
        combined_loaders = {
            "labeled": train_loader,
            "unlabeled": unlabeled_loader
        }

        # 3. Instantiate Lightning Module
        lightning_module = SelfTrainingLightningModule(
            model=self.model,
            config=self.config,
            criterion=self.criterion,
        )

        # 4. Instantiate custom callbacks
        callbacks: list[pl.Callback] = [DashboardTelemetryCallback(self)]

        # 5. Configure PL Trainer options
        accelerator = (
            "gpu" if torch.cuda.is_available() and self.device.type == "cuda" else "cpu"
        )
        devices: Any = 1
        if self.device.type == "cuda" and self.device.index is not None:
            devices = [self.device.index]

        strategy: Any = "auto"
        if self.world_size > 1:
            strategy = "ddp"
            devices = self.world_size

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=callbacks,
            enable_checkpointing=False,  # Checked and saved atomically
            logger=False,  # Handled via local database logging
            enable_progress_bar=self.is_main,
        )

        if self.is_main:
            print("[SelfTrainingTrainer] Starting PyTorch Lightning self-training loop...")
            print(
                f"[SelfTrainingTrainer] Accelerator: {accelerator}, Devices: {devices}, Strategy: {strategy}"
            )

        # Start training
        trainer.fit(lightning_module, combined_loaders, val_loader)
