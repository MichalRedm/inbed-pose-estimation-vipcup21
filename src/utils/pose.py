import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Any

# LSP-style joint connections for skeletal visualization
LSP_SKELETON: List[Tuple[int, int]] = [
    (13, 12),  # Head to Thorax
    (12, 8),
    (12, 9),  # Thorax to Shoulders
    (8, 7),
    (7, 6),  # Right Arm (RS -> RE -> RW)
    (9, 10),
    (10, 11),  # Left Arm (LS -> LE -> LW)
    (8, 2),
    (9, 3),  # Torso (RS -> RH, LS -> LH)
    (2, 3),  # Pelvis (RH -> LH)
    (2, 1),
    (1, 0),  # Right Leg (RH -> RK -> RA)
    (3, 4),
    (4, 5),  # Left Leg (LH -> LK -> LA)
]

# Joint names for debugging/legend
LSP_JOINT_NAMES: List[str] = [
    "R_Ankle",
    "R_Knee",
    "R_Hip",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "R_Wrist",
    "R_Elbow",
    "R_Shoulder",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "Thorax",
    "Head",
]


def decode_heatmaps(
    heatmaps: Union[torch.Tensor, np.ndarray],
    image_size: Tuple[int, int],
    method: str = "argmax",
    temperature: float = 10.0,
) -> torch.Tensor:
    """
    Convert heatmaps (B, J, H, W) to joint coordinates (B, J, 2) in image space.

    Methods:
      - "argmax": Standard peak detection (fast, sensitive to noise)
      - "soft-argmax": Expected value / Center of mass (precise, robust)
    """
    if isinstance(heatmaps, np.ndarray):
        heatmaps_torch = torch.from_numpy(heatmaps)
    else:
        heatmaps_torch = heatmaps

    B, J, H, W = heatmaps_torch.shape
    img_h, img_w = image_size
    device = heatmaps_torch.device

    if method == "soft-argmax":
        # Apply temperature-scaled softmax to get probability distribution
        flat = heatmaps_torch.view(B, J, -1)
        probs = torch.softmax(flat * temperature, dim=-1)
        probs = probs.view(B, J, H, W)

        # Coordinate grids
        grid_x = torch.arange(W, device=device).float().view(1, 1, 1, W)
        grid_y = torch.arange(H, device=device).float().view(1, 1, H, 1)

        # Expected values (center of mass)
        expected_x = torch.sum(probs * grid_x, dim=(2, 3))
        expected_y = torch.sum(probs * grid_y, dim=(2, 3))

        x = expected_x * (img_w / W)
        y = expected_y * (img_h / H)
        return torch.stack([x, y], dim=-1)
    else:
        # Standard argmax
        flat = heatmaps_torch.view(B, J, -1)
        idx = flat.argmax(dim=-1)
        y = (idx // W).float()
        x = (idx % W).float()

        x = x * (img_w / W)
        y = y * (img_h / H)
        return torch.stack([x, y], dim=-1)


def draw_pose(
    ax: Any,
    joints: Union[torch.Tensor, np.ndarray],
    visibility: Optional[Union[torch.Tensor, np.ndarray]] = None,
    color: str = "red",
    linestyle: str = "-",
    label: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    """
    Draw joints and skeletal connections on a Matplotlib axis.
    joints: (J, 2) array or tensor of [x, y] coordinates.
    visibility: (J,) array or tensor of visibility flags (1 or True: draw, 0 or False: skip).
    """
    if torch.is_tensor(joints):
        joints_np = joints.cpu().numpy()
    else:
        joints_np = joints

    if visibility is not None:
        if torch.is_tensor(visibility):
            visibility_np = visibility.cpu().numpy()
        else:
            visibility_np = visibility
    else:
        # Default to all visible if not provided
        visibility_np = np.ones(len(joints_np))

    # Mask for valid joints
    is_visible = visibility_np.astype(bool)

    # Draw connections
    line_drawn = False
    for i, (j1, j2) in enumerate(LSP_SKELETON):
        if j1 < len(joints_np) and j2 < len(joints_np):
            # Only draw if both joints are visible
            if is_visible[j1] and is_visible[j2]:
                x = [joints_np[j1, 0], joints_np[j2, 0]]
                y = [joints_np[j1, 1], joints_np[j2, 1]]

                # Only add label to the first actual line drawn
                plot_label = label if not line_drawn else None
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=plot_label,
                    alpha=alpha,
                )
                line_drawn = True

    # Draw joint points (only visible ones)
    if np.any(is_visible):
        ax.scatter(
            joints_np[is_visible, 0],
            joints_np[is_visible, 1],
            color=color,
            s=20,
            edgecolors="white",
            zorder=5,
            alpha=alpha,
        )


def compute_mpjpe(
    preds: torch.Tensor,
    gts: Union[torch.Tensor, np.ndarray],
    visibility: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Mean Per Joint Position Error.
    preds: (B, J, 2)
    gts: (B, J, 2)
    visibility: (B, J) or (B, 3, J) - if (B, 3, J), index 2 is used as visibility.
    Returns: Average error per joint, and per-joint errors.
    """
    device = preds.device
    if not torch.is_tensor(gts):
        gt_tensor = torch.from_numpy(gts).to(device)
    else:
        gt_tensor = gts.to(device)

    if visibility is not None:
        if not torch.is_tensor(visibility):
            vis_tensor = torch.from_numpy(visibility).to(device)
        else:
            vis_tensor = visibility.to(device)

        if len(vis_tensor.shape) == 3:  # (B, 3, J)
            vis_mask = (vis_tensor[:, 2, :] <= 1).float()
        else:
            vis_mask = vis_tensor
    else:
        vis_mask = torch.ones(preds.shape[:2], device=device)

    # Distance between preds and gts
    dist = torch.sqrt(torch.sum((preds - gt_tensor) ** 2, dim=-1))  # (B, J)

    # Apply visibility mask
    dist = dist * vis_mask

    sum_vis = torch.sum(vis_mask, dim=0)
    per_joint_error = torch.sum(dist, dim=0) / torch.clamp(sum_vis, min=1e-6)

    total_vis = torch.sum(vis_mask)
    mean_error = torch.sum(dist) / torch.clamp(total_vis, min=1e-6)

    return mean_error, per_joint_error


def compute_pck(
    preds: torch.Tensor,
    gts: Union[torch.Tensor, np.ndarray],
    visibility: Optional[Union[torch.Tensor, np.ndarray]] = None,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Percentage of Correct Keypoints.
    threshold: If < 1.0, it's relative to torso diameter (PCK@threshold).
              If >= 1.0, it's absolute pixel threshold.
    """
    device = preds.device
    if not torch.is_tensor(gts):
        gt_tensor = torch.from_numpy(gts).to(device)
    else:
        gt_tensor = gts.to(device)

    if visibility is not None:
        if not torch.is_tensor(visibility):
            vis_tensor = torch.from_numpy(visibility).to(device)
        else:
            vis_tensor = visibility.to(device)

        if len(vis_tensor.shape) == 3:  # (B, 3, J)
            vis_mask = (vis_tensor[:, 2, :] <= 1).float()
        else:
            vis_mask = vis_tensor
    else:
        vis_mask = torch.ones(preds.shape[:2], device=device)

    dist = torch.sqrt(torch.sum((preds - gt_tensor) ** 2, dim=-1))  # (B, J)

    if threshold < 1.0:
        # Relative threshold (PCK@threshold)
        # Use distance between midpoint of shoulders and midpoint of hips as torso reference
        # Indices: 8:RShoulder, 9:LShoulder, 2:RHip, 3:LHip
        shoulder_mid = (gt_tensor[:, 8, :] + gt_tensor[:, 9, :]) / 2.0
        hip_mid = (gt_tensor[:, 2, :] + gt_tensor[:, 3, :]) / 2.0
        torso_dist = torch.sqrt(
            torch.sum((shoulder_mid - hip_mid) ** 2, dim=-1)
        )  # (B,)
        # Ensure minimum torso distance to avoid division by zero
        torso_dist = torch.clamp(torso_dist, min=1e-6)

        # Reshape torso_dist to (B, 1) for broadcasting
        effective_threshold = torso_dist.unsqueeze(-1) * threshold
    else:
        effective_threshold = torch.tensor(threshold, device=device)

    correct = (dist <= effective_threshold).float() * vis_mask

    sum_vis = torch.sum(vis_mask, dim=0)
    per_joint_pck = torch.sum(correct, dim=0) / torch.clamp(sum_vis, min=1e-6)

    total_vis = torch.sum(vis_mask)
    mean_pck = torch.sum(correct) / torch.clamp(total_vis, min=1e-6)

    return mean_pck, per_joint_pck
