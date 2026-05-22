import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.models.hrnet import HRNet
from src.training.losses import FeatureDistillationLoss, UncertaintyWeighting
from src.training.factory import create_trainer
from src.training.distillation_trainer import DistillationTrainer
from src.data.dataset import VIPCupDataset

def test_hrnet_return_stages():
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {
                "num_joints": 14,
                "in_channels": 3,
                "pretrained": False
            }
        }
    }
    model = HRNet(config)
    x = torch.randn(2, 3, 256, 256)
    
    # Test standard output
    out = model(x)
    assert out.shape == (2, 14, 64, 64)
    
    # Test stages output
    out, stage3, stage4 = model(x, return_stages=True)
    assert out.shape == (2, 14, 64, 64)
    assert stage3.shape == (2, 224, 64, 64)
    assert stage4.shape == (2, 480, 64, 64)


def test_feature_distillation_loss():
    # Matching channels
    loss_fn = FeatureDistillationLoss(channels_student=[224, 480], channels_teacher=[224, 480])
    
    s_s3 = torch.randn(2, 224, 64, 64)
    s_s4 = torch.randn(2, 480, 64, 64)
    t_s3 = torch.randn(2, 224, 64, 64)
    t_s4 = torch.randn(2, 480, 64, 64)
    
    loss = loss_fn([s_s3, s_s4], [t_s3, t_s4])
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    
    # Mismatched channels (should apply projection)
    loss_fn_mismatch = FeatureDistillationLoss(channels_student=[128, 256], channels_teacher=[224, 480])
    s_m3 = torch.randn(2, 128, 64, 64)
    s_m4 = torch.randn(2, 256, 64, 64)
    
    loss_mismatch = loss_fn_mismatch([s_m3, s_m4], [t_s3, t_s4])
    assert isinstance(loss_mismatch, torch.Tensor)
    assert loss_mismatch.item() >= 0


def test_uncertainty_weighting():
    uw = UncertaintyWeighting(num_tasks=2)
    losses = {
        "pose": torch.tensor(1.5, requires_grad=True),
        "distill": torch.tensor(0.8, requires_grad=True)
    }
    total_loss, weighted_dict = uw(losses)
    assert isinstance(total_loss, torch.Tensor)
    assert "w_pose" in weighted_dict
    assert "w_distill" in weighted_dict
    assert "sigma_pose" in weighted_dict
    assert "sigma_distill" in weighted_dict
    
    # Test backward pass propagates gradients
    total_loss.backward()
    assert losses["pose"].grad is not None
    assert losses["distill"].grad is not None
    assert uw.log_vars.grad is not None


def test_distillation_trainer_creation(tmp_path):
    # Create a mock checkpoint file
    teacher_path = tmp_path / "rgb_teacher.pth"
    dummy_model = HRNet({
        "in_channels": 3,
        "pretrained": False,
        "num_joints": 14
    })
    torch.save(dummy_model.state_dict(), teacher_path)
    
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {
                "num_joints": 14,
                "in_channels": 1,
                "pretrained": False
            }
        },
        "training": {
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "save_dir": str(tmp_path)
        },
        "dataset": {
            "image_size": [256, 256]
        },
        "training_type": "distillation",
        "distillation": {
            "teacher_checkpoint": str(teacher_path),
            "use_uncertainty_weighting": True
        }
    }
    
    device = torch.device("cpu")
    trainer, model = create_trainer(config, device)
    
    assert isinstance(trainer, DistillationTrainer)
    assert isinstance(model, nn.Module)
    assert trainer.device == device
    assert trainer.teacher is not None
    
    # Run a mock single step
    batch = {
        "image": torch.randn(2, 1, 256, 256),
        "joints": torch.randn(2, 3, 14),
        "image_aligned": torch.randn(2, 3, 256, 256)
    }
    
    metrics = trainer._train_step(batch)
    assert "loss" in metrics
    assert "loss_pose" in metrics
    assert "loss_distill" in metrics
    assert "w_pose" in metrics
    assert "w_distill" in metrics
