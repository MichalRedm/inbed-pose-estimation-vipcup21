import torch
import pytest
from src.training.losses import AnatomicalLoss


def test_two_sided_hinge_loss():
    """Verify that two_sided_hinge mode penalizes both excessive stretching and excessive shrinking."""
    # 1 batch, 14 joints, 2D coordinates (scaled to 256)
    joints_perfect = torch.zeros((1, 14, 2))

    # Construct a skeleton with exact target lengths
    # We can instantiate the loss to get the priors list
    loss_fn = AnatomicalLoss(mode="two_sided_hinge", image_size=256)
    
    # Set joints based on target lengths to have zero/low loss
    # LSP indices:
    # 0:R_Ankle, 1:R_Knee, 2:R_Hip, 3:L_Hip, 4:L_Knee, 5:L_Ankle
    # 6:R_Wrist, 7:R_Elbow, 8:R_Shoulder, 9:L_Shoulder, 10:L_Elbow, 11:L_Wrist
    # 12:Thorax, 13:Head
    
    # Setup coordinates (each joint spaced exactly by target length in Y-axis)
    # R_Lower_Leg (0, 1) target = 49
    joints_perfect[0, 1] = torch.tensor([50.0, 50.0])
    joints_perfect[0, 0] = torch.tensor([50.0, 50.0 + 49.0])
    
    # R_Upper_Leg (1, 2) target = 54
    joints_perfect[0, 2] = torch.tensor([50.0, 50.0 - 54.0])

    # L_Lower_Leg (5, 4) target = 51
    joints_perfect[0, 4] = torch.tensor([100.0, 50.0])
    joints_perfect[0, 5] = torch.tensor([100.0, 50.0 + 51.0])

    # L_Upper_Leg (4, 3) target = 52
    joints_perfect[0, 3] = torch.tensor([100.0, 50.0 - 52.0])

    # Run loss on perfect/neutral skeleton
    loss_perfect = loss_fn(joints_perfect)
    
    # Let's create a collapsed skeleton (where lower legs shrink to 0 length)
    joints_collapsed = joints_perfect.clone()
    joints_collapsed[0, 0] = joints_collapsed[0, 1]  # R Lower Leg length = 0
    joints_collapsed[0, 5] = joints_collapsed[0, 4]  # L Lower Leg length = 0
    
    loss_collapsed = loss_fn(joints_collapsed)
    
    # Collapsed skeleton should have a MUCH higher loss than perfect skeleton
    assert loss_collapsed > loss_perfect
    assert loss_collapsed.item() > 0.0

    # Let's create an excessively stretched skeleton (where R Lower Leg length = 150)
    joints_stretched = joints_perfect.clone()
    joints_stretched[0, 0] = torch.tensor([50.0, 50.0 + 150.0])
    
    loss_stretched = loss_fn(joints_stretched)
    assert loss_stretched > loss_perfect
    assert loss_stretched.item() > 0.0
