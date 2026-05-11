import torch
import pytest
from src.utils.pose import compute_pck, compute_mpjpe


def test_pck_vis_masking():
    """Verify that PCK correctly handles visible vs occluded vs missing joints."""
    # 14 joints
    preds = torch.zeros((1, 14, 2))
    gts = torch.zeros((1, 14, 2))

    # 3x14 visibility: [x, y, vis]
    # vis: 0=visible, 1=occluded, 2=missing
    visibility = torch.full((1, 3, 14), 2.0)

    # Set all joints to (0,0)
    # Joint 0: visible, Correct
    visibility[0, 2, 0] = 0

    # Joint 1: occluded, Correct
    visibility[0, 2, 1] = 1

    # Joint 2: missing, Correct (but should be ignored)
    visibility[0, 2, 2] = 2

    # Joint 3: visible, Incorrect (far away)
    visibility[0, 2, 3] = 0
    preds[0, 3, :] = 100

    # Joint 4: occluded, Incorrect (far away)
    visibility[0, 2, 4] = 1
    preds[0, 4, :] = 100

    # Torso distance: shoulder-mid (8,9) to hip-mid (2,3)
    # Set them so torso distance is 100
    gts[0, 8, :] = torch.tensor([0, 0])  # R Shoulder
    gts[0, 9, :] = torch.tensor([0, 0])  # L Shoulder
    gts[0, 2, :] = torch.tensor([0, 100])  # R Hip
    gts[0, 3, :] = torch.tensor([0, 100])  # L Hip

    # Mean PCK should consider joints 0, 1, 3, 4 (count=4)
    # Correct: 0, 1 (count=2)
    # PCK = 2/4 = 0.5
    mean_pck, per_joint_pck = compute_pck(
        preds, gts, visibility=visibility, threshold=0.5
    )

    assert mean_pck == 0.5
    assert per_joint_pck[0] == 1.0
    assert per_joint_pck[1] == 1.0
    assert (
        per_joint_pck[2] == 0.0
    )  # Missing joint count is 1 in pose.py's per_joint_pck due to clamp, but masked in mean
    assert per_joint_pck[3] == 0.0
    assert per_joint_pck[4] == 0.0


def test_mpjpe_consistency():
    """Verify MPJPE calculation and visibility masking."""
    preds = torch.zeros((1, 14, 2))
    gts = torch.zeros((1, 14, 2))
    visibility = torch.full((1, 3, 14), 2.0)

    # Joint 0: visible, error = 10
    visibility[0, 2, 0] = 0
    preds[0, 0, 0] = 10

    # Joint 1: occluded, error = 20
    visibility[0, 2, 1] = 1
    preds[0, 1, 0] = 20

    # Joint 2: missing, error = 100 (should be ignored)
    visibility[0, 2, 2] = 2
    preds[0, 2, 0] = 100

    # Mean error = (10 + 20) / 2 = 15
    mean_err, per_joint_err = compute_mpjpe(preds, gts, visibility=visibility)

    assert mean_err == 15.0
    assert per_joint_err[0] == 10.0
    assert per_joint_err[1] == 20.0


if __name__ == "__main__":
    pytest.main([__file__])
