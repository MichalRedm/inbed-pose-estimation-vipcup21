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
    SLP dataset semantics:
    - 0: visible
    - 1: occluded (e.g., under cover) -> We WANT heatmaps for these to train model to 'see' through blankets.
    - 2: out of view / missing -> No heatmaps.
    """
    dataset = _make_dataset()

    # 1. Visible joints (if_occluded == 0)
    visible_joints = torch.zeros((3, 14))
    visible_joints[0, :] = 64
    visible_joints[1, :] = 64
    visible_joints[2, :] = 0
    hm_visible = dataset._generate_heatmaps(visible_joints)
    assert hm_visible.sum() > 0, "Visible joints must produce heatmaps"

    # 2. Occluded joints (if_occluded == 1)
    # In VIP Cup, we predict joints even under blankets
    occluded_joints = torch.zeros((3, 14))
    occluded_joints[0, :] = 64
    occluded_joints[1, :] = 64
    occluded_joints[2, :] = 1
    hm_occluded = dataset._generate_heatmaps(occluded_joints)
    assert hm_occluded.sum() > 0, (
        "Occluded joints (under cover) must produce heatmaps for training"
    )

    # 3. Out of view joints (if_occluded == 2)
    missing_joints = torch.zeros((3, 14))
    missing_joints[0, :] = 64
    missing_joints[1, :] = 64
    missing_joints[2, :] = 2
    hm_missing = dataset._generate_heatmaps(missing_joints)
    assert hm_missing.sum() == 0, (
        "Missing/Out-of-view joints must produce all-zero heatmaps"
    )


def test_diverse_sigmas():
    """Verify that heatmap generation is stable across various sigma values (curriculum decay)."""
    dataset = _make_dataset()
    joints = torch.zeros((3, 14))
    joints[0, 0] = 128
    joints[1, 0] = 128
    joints[2, 0] = 0

    # Test all sigma values likely to occur during curriculum scheduling
    for sigma in [3.0, 2.5, 2.0, 1.75, 1.6, 1.5, 1.0, 0.5]:
        dataset.set_sigma(sigma)
        try:
            hm = dataset._generate_heatmaps(joints)
            assert hm.shape == (14, 64, 64), f"Wrong shape for sigma={sigma}"
            assert hm[0].max() > 0, f"Gaussian peak missing for sigma={sigma}"
        except Exception as e:
            pytest.fail(f"Heatmap generation failed with sigma={sigma}: {e}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
