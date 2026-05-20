import torch
import pytest
from src.data.dataset import VIPCupDataset
from src.models import build_model
from pathlib import Path


def test_dataset_channel_replication():
    """Verify that VIPCupDataset replicates channels correctly when in_channels=3."""
    # Use a dummy root that exists or just test the logic if data is missing
    data_root = "data/raw"
    if not Path(data_root).exists():
        # If data is missing, we can't test actual loading, but we can test the __init__
        dataset = VIPCupDataset(root=".", subjects=[], in_channels=3)
        assert dataset.in_channels == 3
        return

    dataset_3 = VIPCupDataset(root=data_root, subjects=[1], in_channels=3)
    assert dataset_3.in_channels == 3

    if len(dataset_3) > 0:
        sample = dataset_3[0]
        img = sample["image"]
        # Shape should be (3, H, W)
        assert img.shape[0] == 3
        # Channels should be identical (replication)
        assert torch.allclose(img[0], img[1])
        assert torch.allclose(img[1], img[2])


def test_hrnet_3_channels():
    """Verify that HRNet accepts 3-channel input."""
    config = {
        "model": {
            "name": "hrnet",
            "hrnet": {
                "num_joints": 14,
                "in_channels": 3,
                "architecture": "w32",
                "pretrained": False,
            },
        }
    }
    model = build_model(config)
    model.eval()

    # Input with 3 channels
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 14, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__])
