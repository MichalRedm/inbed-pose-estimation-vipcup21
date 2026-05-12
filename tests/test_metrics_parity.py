import torch
import numpy as np
import pytest
from src.utils.pose import (
    compute_pck as new_compute_pck,
    compute_mpjpe as new_compute_mpjpe,
)


# --- LEGACY IMPLEMENTATION (Numpy-based) ---
def legacy_compute_mpjpe(preds, gts, visibility=None):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(gts):
        gts = gts.cpu().numpy()

    if visibility is not None:
        if torch.is_tensor(visibility):
            visibility = visibility.cpu().numpy()
        if len(visibility.shape) == 3:  # (B, 3, J)
            visibility = (visibility[:, 2, :] <= 1).astype(float)
    else:
        visibility = np.ones(preds.shape[:2])

    dist = np.sqrt(np.sum((preds - gts) ** 2, axis=-1))
    dist = dist * visibility
    per_joint_error = np.sum(dist, axis=0) / np.maximum(np.sum(visibility, axis=0), 1)
    mean_error = np.sum(dist) / np.maximum(np.sum(visibility), 1)
    return mean_error, per_joint_error


def legacy_compute_pck(preds, gts, visibility=None, threshold=0.5):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(gts):
        gts = gts.cpu().numpy()

    if visibility is not None:
        if torch.is_tensor(visibility):
            visibility = visibility.cpu().numpy()
        if len(visibility.shape) == 3:  # (B, 3, J)
            visibility = (visibility[:, 2, :] <= 1).astype(float)
    else:
        visibility = np.ones(preds.shape[:2])

    dist = np.sqrt(np.sum((preds - gts) ** 2, axis=-1))

    if threshold < 1.0:
        shoulder_mid = (gts[:, 8, :] + gts[:, 9, :]) / 2.0
        hip_mid = (gts[:, 2, :] + gts[:, 3, :]) / 2.0
        torso_dist = np.sqrt(np.sum((shoulder_mid - hip_mid) ** 2, axis=-1))
        torso_dist = np.maximum(torso_dist, 1e-6)
        effective_threshold = torso_dist[:, np.newaxis] * threshold
    else:
        effective_threshold = threshold

    correct = (dist <= effective_threshold).astype(float) * visibility
    per_joint_pck = np.sum(correct, axis=0) / np.maximum(np.sum(visibility, axis=0), 1)
    mean_pck = np.sum(correct) / np.maximum(np.sum(visibility), 1)
    return mean_pck, per_joint_pck


# --- PARITY TESTS ---
@pytest.mark.parametrize("batch_size", [1, 4])
def test_mpjpe_parity(batch_size):
    torch.manual_seed(42)
    preds = torch.randn(batch_size, 14, 2) * 100
    gts = torch.randn(batch_size, 14, 2) * 100
    visibility = torch.randint(0, 3, (batch_size, 3, 14)).float()

    legacy_mean, legacy_per_joint = legacy_compute_mpjpe(preds, gts, visibility)
    new_mean, new_per_joint = new_compute_mpjpe(preds, gts, visibility)

    assert np.allclose(
        legacy_mean,
        new_mean.cpu().numpy() if torch.is_tensor(new_mean) else new_mean,
        atol=1e-5,
    )
    assert np.allclose(
        legacy_per_joint,
        new_per_joint.cpu().numpy()
        if torch.is_tensor(new_per_joint)
        else new_per_joint,
        atol=1e-5,
    )


@pytest.mark.parametrize("threshold", [0.2, 0.5, 2.0])
def test_pck_parity(threshold):
    batch_size = 4
    torch.manual_seed(42)
    preds = torch.randn(batch_size, 14, 2) * 100
    gts = torch.randn(batch_size, 14, 2) * 100
    # Ensure some torso distance
    gts[:, 8, :] = 0
    gts[:, 9, :] = 10
    gts[:, 2, :] = 0
    gts[:, 3, :] = 50
    visibility = torch.randint(0, 3, (batch_size, 3, 14)).float()

    legacy_mean, legacy_per_joint = legacy_compute_pck(
        preds, gts, visibility, threshold
    )
    new_mean, new_per_joint = new_compute_pck(preds, gts, visibility, threshold)

    assert np.allclose(
        legacy_mean,
        new_mean.cpu().numpy() if torch.is_tensor(new_mean) else new_mean,
        atol=1e-5,
    )
    assert np.allclose(
        legacy_per_joint,
        new_per_joint.cpu().numpy()
        if torch.is_tensor(new_per_joint)
        else new_per_joint,
        atol=1e-5,
    )
