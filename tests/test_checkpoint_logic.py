import torch
import pytest
import os
import shutil
from src.training.base_trainer import BaseTrainer
from typing import Dict, Any


class MockTrainer(BaseTrainer):
    """Minimal implementation of BaseTrainer for testing."""

    def __init__(self, save_dir):
        self.config = {"run_id": "test_run"}
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.best_val_pck = 0.0
        self.best_val_loss = float("inf")
        self.save_dir = save_dir
        self.is_main = True
        self.world_size = 1

        # Create checkpoints dir
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    def train_epoch(self, epoch):
        return 0.0

    def validate(self, epoch):
        return 0.0

    def _train_step(self, batch):
        return 0.0

    def _val_step(self, batch):
        return 0.0

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {}


def test_best_checkpoint_selection():
    """Verify the logic that determines is_best based on val_pck."""
    save_dir = "temp_ckpt"
    trainer = MockTrainer(save_dir)

    # Simulate first epoch
    val_pck = 0.75
    val_loss = 0.02

    # is_best logic from StandardTrainer:
    is_best = val_pck > trainer.best_val_pck
    if is_best:
        trainer.best_val_pck = val_pck
    if val_loss < trainer.best_val_loss:
        trainer.best_val_loss = val_loss

    assert is_best
    trainer.save_checkpoint("epoch_1", is_best=is_best)
    assert os.path.exists(os.path.join(save_dir, "checkpoints", "best_model.pth"))

    # Simulate second epoch: Better loss, worse PCK
    val_pck = 0.72
    val_loss = 0.01

    is_best = val_pck > trainer.best_val_pck
    if is_best:
        trainer.best_val_pck = val_pck
    if val_loss < trainer.best_val_loss:
        trainer.best_val_loss = val_loss

    assert not is_best, "Lower PCK should NOT be is_best"
    assert trainer.best_val_loss == 0.01, "Loss should still be updated"

    # Cleanup
    shutil.rmtree(save_dir)


if __name__ == "__main__":
    pytest.main([__file__])
