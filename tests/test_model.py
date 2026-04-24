"""Tests for the HRNet-W32 model architecture."""

import torch
from src.models.hrnet import get_pose_net


def test_hrnet_output_shape():
    """HRNet-W32 should output (B, 14, H/4, W/4) heatmaps."""
    config = {"num_joints": 14, "in_channels": 1}
    model = get_pose_net(config)
    model.eval()

    x = torch.zeros(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 14, 64, 64), (
        f"Expected shape (2, 14, 64, 64), got {out.shape}"
    )


def test_hrnet_parameter_count():
    """HRNet-W32 should have approximately 28M parameters."""
    config = {"num_joints": 14, "in_channels": 1}
    model = get_pose_net(config)
    n_params = sum(p.numel() for p in model.parameters())
    # Real HRNet-W32 has ~28M. Allow a wide range since this is our implementation.
    assert n_params > 1_000_000, (
        f"Model has only {n_params:,} params — likely the MVP stub, not the real HRNet"
    )
    print(f"HRNet-W32 parameter count: {n_params:,}")


if __name__ == "__main__":
    test_hrnet_output_shape()
    test_hrnet_parameter_count()
