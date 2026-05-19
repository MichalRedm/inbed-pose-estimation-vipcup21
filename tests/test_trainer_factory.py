import pytest
import torch
import torch.nn as nn
from src.training.factory import create_trainer
from src.training.standard_trainer import StandardTrainer
from src.training.uda_trainer import UDATrainer


@pytest.fixture
def base_config(tmp_path):
    return {
        "model": {"name": "hrnet", "hrnet": {"num_joints": 14, "in_channels": 1}},
        "training": {
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "save_dir": str(tmp_path / "models" / "checkpoints"),
        },
        "dataset": {"image_size": [256, 256]},
    }


def test_create_standard_trainer(base_config):
    device = torch.device("cpu")
    trainer, model = create_trainer(base_config, device)

    assert isinstance(trainer, StandardTrainer)
    assert isinstance(model, nn.Module)
    assert trainer.device == device


def test_create_uda_trainer(base_config):
    base_config["training_type"] = "uda"
    base_config["uda"] = {"enabled": True, "lambda_adv": 0.1}

    device = torch.device("cpu")
    trainer, model = create_trainer(base_config, device)

    assert isinstance(trainer, UDATrainer)
    assert hasattr(trainer, "discriminator")
    assert trainer.lambda_adv == 0.1


def test_anatomical_constraints_config(base_config):
    base_config["training"]["lambda_anatomical"] = 0.5
    device = torch.device("cpu")
    trainer, model = create_trainer(base_config, device)

    # StandardTrainer should have lambda_anatomical if it uses it
    # We check if it's passed through the config
    assert trainer.config["training"]["lambda_anatomical"] == 0.5


def test_factory_invalid_config():
    # Test fallback or error handling if needed
    config = {"model": {"name": "invalid"}}
    with pytest.raises(Exception):
        create_trainer(config, torch.device("cpu"))


def test_discriminative_lr(base_config):
    base_config["training"]["backbone_lr_ratio"] = 0.1
    device = torch.device("cpu")
    trainer, model = create_trainer(base_config, device)

    # The optimizer should have exactly 2 parameter groups
    optimizer = trainer.optimizer
    assert len(optimizer.param_groups) == 2

    # Identify the learning rates in the parameter groups
    lrs = [group["lr"] for group in optimizer.param_groups]

    # One learning rate should be the full head learning rate (0.0001)
    # The other should be the backbone learning rate (0.0001 * 0.1 = 0.00001)
    assert 0.0001 in lrs
    assert 0.00001 in lrs
