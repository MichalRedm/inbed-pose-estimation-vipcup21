import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AnatomicalLoss(nn.Module):
    """
    Enforces physical consistency of the predicted skeleton.
    Includes:
    1. Bone length prior loss (L2 distance to mean dataset lengths)
    2. Symmetry loss (L/R limb length mismatch penalty)
    """

    def __init__(self, device="cpu", image_size=256, mode="hinge"):
        """
        Anatomical Constraint Loss for pose estimation.

        Args:
            device: Device to store priors on.
            image_size: Original image dimensions for normalization.
            mode: 'mse' for fixed bone lengths, 'hinge' for upper-bound only (allows foreshortening).
        """
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.mode = mode

        # Skeleton definition (LSP indices)
        # 0:R_Ankle, 1:R_Knee, 2:R_Hip, 3:L_Hip, 4:L_Knee, 5:L_Ankle,
        # 6:R_Wrist, 7:R_Elbow, 8:R_Shoulder, 9:L_Shoulder, 10:L_Elbow, 11:L_Wrist,
        # 12:Thorax, 13:Head

        # Bones with reliable priors (normalized to 0-1 by dividing by 256)
        # Format: (j1, j2, target_length_normalized)
        self.priors = [
            (0, 1, 49.0 / 256.0),  # R_Lower_Leg
            (1, 2, 54.0 / 256.0),  # R_Upper_Leg
            (5, 4, 51.0 / 256.0),  # L_Lower_Leg
            (4, 3, 52.0 / 256.0),  # L_Upper_Leg
            (6, 7, 27.0 / 256.0),  # R_Forearm
            (7, 8, 38.0 / 256.0),  # R_Upper_Arm
            (11, 10, 27.0 / 256.0),  # L_Forearm
            (10, 9, 40.0 / 256.0),  # L_Upper_Arm
            (12, 13, 25.0 / 256.0),  # Neck/Head
        ]

        # Symmetrical pairs to enforce length equality
        self.symmetrical_pairs = [
            ((5, 4), (0, 1)),  # Lower Legs
            ((4, 3), (1, 2)),  # Upper Legs
            ((11, 10), (6, 7)),  # Forearms
            ((10, 9), (7, 8)),  # Upper Arms
            ((3, 12), (2, 12)),  # Torso sides (Hip to Thorax)
        ]

    def forward(self, pred_joints):
        """
        pred_joints: (B, 14, 2) - Joint coordinates in image space (0-256)
        """
        # Normalize predicted joints to 0-1
        pred_joints_norm = pred_joints / self.image_size

        loss_prior = 0.0
        loss_sym = 0.0

        # 1. Bone Length Prior Loss
        for j1, j2, target in self.priors:
            p1 = pred_joints_norm[:, j1]
            p2 = pred_joints_norm[:, j2]
            length = torch.norm(p1 - p2, dim=1)

            if self.mode == "hinge":
                # Use squared ReLU for a smooth hinge loss (penalize only if length > target)
                loss_prior += torch.mean(torch.pow(F.relu(length - target), 2))
            else:
                # MSE: Force length to match target exactly
                loss_prior += F.mse_loss(length, torch.full_like(length, target))

        # 2. Symmetry Loss (Keep MSE)
        for (lj1, lj2), (rj1, rj2) in self.symmetrical_pairs:
            lp1 = pred_joints_norm[:, lj1]
            lp2 = pred_joints_norm[:, lj2]
            rp1 = pred_joints_norm[:, rj1]
            rp2 = pred_joints_norm[:, rj2]

            l_len = torch.norm(lp1 - lp2, dim=1)
            r_len = torch.norm(rp1 - rp2, dim=1)
            loss_sym += F.mse_loss(l_len, r_len)

        return (loss_prior / len(self.priors)) + (
            loss_sym / len(self.symmetrical_pairs)
        )


class UncertaintyWeighting(nn.Module):
    """
    Implements multi-task loss weighting using learned uncertainties.
    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        # Initial log-variances set to 0 (sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        losses: Dictionary of individual losses.
        Returns: (total_loss, weighted_losses_dict)
        """
        total_loss = 0
        weighted_dict = {}

        # We need a stable order for the log_vars. 
        # We'll use the sorted keys of the input dictionary.
        keys = sorted(losses.keys())
        for i, key in enumerate(keys):
            loss = losses[key]
            # L = exp(-s) * loss + s
            # s = log(sigma^2)
            log_var = self.log_vars[i]
            weighted_loss = torch.exp(-log_var) * loss + log_var
            
            total_loss += weighted_loss
            weighted_dict[f"w_{key}"] = weighted_loss.item()
            weighted_dict[f"sigma_{key}"] = torch.exp(0.5 * log_var).item()

        return total_loss, weighted_dict
