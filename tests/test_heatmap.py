import torch
from src.data.dataset import VIPCupDataset


def _make_dataset():
    dataset = VIPCupDataset(root=".", subjects=[], image_size=(256, 256))
    dataset.heatmap_size = (64, 64)
    dataset.sigma = 2.0
    return dataset


def test_heatmap_generation():
    """Visible joint at image centre should produce a Gaussian peak in the heatmap."""
    joints = torch.zeros((3, 14))
    joints[0, 0] = 128  # centre x
    joints[1, 0] = 128  # centre y
    joints[2, 0] = 0  # visible (if_occluded == 0)

    dataset = _make_dataset()
    heatmaps = dataset._generate_heatmaps(joints)

    assert heatmaps.shape == (14, 64, 64)
    # Joint 0 at (128,128) in 256×256 → (32,32) in 64×64
    assert heatmaps[0, 32, 32] > 0.9, "Visible joint should produce a strong peak"
    assert heatmaps[0, 0, 0] < 0.01, "Far corner should be near zero"


def test_visibility_semantics():
    """
    Dataset README: if_occluded == 0 means VISIBLE, != 0 means occluded.
    Visible joints must produce non-zero heatmaps; occluded joints must not.
    """
    dataset = _make_dataset()

    # All visible (if_occluded == 0)
    visible_joints = torch.zeros((3, 14))
    visible_joints[0, :] = 64  # x
    visible_joints[1, :] = 64  # y
    visible_joints[2, :] = 0  # visible
    hm_visible = dataset._generate_heatmaps(visible_joints)
    assert hm_visible.sum() > 0, "Visible joints must produce non-zero heatmaps"

    # All occluded (if_occluded == 1)
    occluded_joints = torch.zeros((3, 14))
    occluded_joints[0, :] = 64
    occluded_joints[1, :] = 64
    occluded_joints[2, :] = 1  # occluded
    hm_occluded = dataset._generate_heatmaps(occluded_joints)
    assert hm_occluded.sum() == 0, "Occluded joints must produce all-zero heatmaps"


if __name__ == "__main__":
    test_heatmap_generation()
    test_visibility_semantics()
