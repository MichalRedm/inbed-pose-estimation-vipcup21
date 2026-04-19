import torch
from src.data.dataset import VIPCupDataset


def test_heatmap_generation():
    # Mock some joints
    # Shape (3, 14) -> x, y, visibility
    joints = torch.zeros((3, 14))
    joints[0, 0] = 128  # center x
    joints[1, 0] = 128  # center y
    joints[2, 0] = 1  # visible

    # Create dataset instance with dummy root
    dataset = VIPCupDataset(root=".", subjects=[], image_size=(256, 256))
    dataset.heatmap_size = (64, 64)
    dataset.sigma = 2.0

    heatmaps = dataset._generate_heatmaps(joints)

    assert heatmaps.shape == (14, 64, 64)

    # The joint at (128, 128) in 256x256 image should be at (32, 32) in 64x64 heatmap
    val = heatmaps[0, 32, 32]
    print(f"Heatmap value at center: {val}")

    assert val > 0.9  # Should be near the max of Gaussian
    assert heatmaps[0, 0, 0] < 0.01  # Should be near zero at corners

    print("Heatmap generation test passed!")


if __name__ == "__main__":
    test_heatmap_generation()
