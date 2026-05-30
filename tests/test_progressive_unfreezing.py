import torch
from typing import Dict, Any, cast
from src.models.hrnet import HRNet
from src.training.factory import create_trainer, build_optimizer
from src.training.standard_trainer import StandardTrainer


def test_hrnet_unfreeze_all() -> None:
    config: Dict[str, Any] = {
        "num_joints": 14,
        "in_channels": 1,
        "pretrained": False,
        "freeze_stem": True,
        "freeze_stage1": True,
    }
    model = HRNet(config)

    # Stem and stage1 parameters should be frozen initially
    assert any(not p.requires_grad for p in model.conv1.parameters())
    assert any(not p.requires_grad for p in model.layer1.parameters())

    # After unfreeze_all, all parameters should have requires_grad=True
    model.unfreeze_all()
    assert all(p.requires_grad for p in model.parameters())


def test_standard_trainer_progressive_unfreezing_setup() -> None:
    config: Dict[str, Any] = {
        "model": {
            "name": "hrnet",
            "hrnet": {
                "num_joints": 14,
                "in_channels": 1,
                "pretrained": False,
                "freeze_stem": True,
                "freeze_stage1": True,
            },
        },
        "training": {
            "lr": 0.0001,
            "backbone_lr_ratio": 0.1,
            "unfreeze_epoch": 5,
            "weight_decay": 0.0001,
            "save_dir": "results/runs/test_progressive_unfreezing",
        },
        "dataset": {"image_size": [256, 256]},
    }

    device = torch.device("cpu")
    trainer, model = create_trainer(config, device)

    assert isinstance(trainer, StandardTrainer)
    assert trainer.unfreeze_epoch == 5
    assert trainer.backbone_lr_ratio == 0.1

    # In Phase 1 (epoch < 5), only unfrozen parameters are in the optimizer (head)
    # Rebuild optimizer mimics Phase 2 (epoch = 5):
    # Unfreeze the model parameters first
    model_hrnet = cast(HRNet, model)
    model_hrnet.unfreeze_all()
    assert all(p.requires_grad for p in model_hrnet.parameters())

    # Rebuild optimizer
    new_opt = build_optimizer(model_hrnet, trainer, config)
    assert len(new_opt.param_groups) == 2

    lrs = [group["lr"] for group in new_opt.param_groups]
    assert 0.0001 in lrs
    assert 0.00001 in lrs

    # Cleanup save_dir
    import shutil
    import os

    save_dir = str(config["training"]["save_dir"])
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
