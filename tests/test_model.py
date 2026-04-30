"""Tests for the model architecture and registry."""

import torch
import pytest
from src.models import build_model, MODEL_REGISTRY
from src.models.base import BaseModel
from src.models.hrnet import HRNet


def test_registry_contains_hrnet():
    """Registry should contain the 'hrnet' architecture."""
    assert "hrnet" in MODEL_REGISTRY
    assert MODEL_REGISTRY["hrnet"] == HRNet


def test_build_model_hrnet():
    """build_model should correctly instantiate HRNet from config."""
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {"num_joints": 14, "in_channels": 1, "architecture": "w32"},
        }
    }
    model = build_model(config)
    assert isinstance(model, HRNet)
    assert isinstance(model, BaseModel)


def test_hrnet_output_shape():
    """HRNet should output (B, 14, H/4, W/4) heatmaps."""
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {"num_joints": 14, "in_channels": 1, "architecture": "w32"},
        }
    }
    model = build_model(config)
    model.eval()

    x = torch.zeros(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 14, 64, 64), (
        f"Expected shape (2, 14, 64, 64), got {out.shape}"
    )


def test_hrnet_parameter_count():
    """HRNet-W32 should have a significant number of parameters."""
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {"num_joints": 14, "in_channels": 1, "architecture": "w32"},
        }
    }
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    # HRNet-W32 typically has > 20M params.
    assert n_params > 1_000_000, (
        f"Model has only {n_params:,} params — likely too small"
    )


def test_build_invalid_model():
    """build_model should raise ValueError for unregistered models."""
    config = {"model": {"name": "invalid_model", "invalid_model": {}}}
    with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
        build_model(config)


if __name__ == "__main__":
    # Allow running directly
    test_registry_contains_hrnet()
    test_build_model_hrnet()
    test_hrnet_output_shape()
    test_hrnet_parameter_count()
    print("All model tests passed!")
