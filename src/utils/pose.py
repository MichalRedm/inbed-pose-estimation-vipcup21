import torch
import numpy as np

# LSP-style joint connections for skeletal visualization
LSP_SKELETON = [
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
LSP_JOINT_NAMES = [
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


def decode_heatmaps(heatmaps, image_size):
    """
    Convert heatmaps (B, J, H, W) to joint coordinates (B, J, 2) in image space.
    Uses argmax followed by upscaling to image resolution.
    """
    if isinstance(heatmaps, np.ndarray):
        heatmaps = torch.from_numpy(heatmaps)

    B, J, H, W = heatmaps.shape
    flat = heatmaps.view(B, J, -1)
    idx = flat.argmax(dim=-1)  # (B, J)
    y = (idx // W).float()
    x = (idx % W).float()

    # Scale to image space
    img_h, img_w = image_size
    x = x * (img_w / W)
    y = y * (img_h / H)

    return torch.stack([x, y], dim=-1)


def draw_pose(
    ax, joints, visibility=None, color="red", linestyle="-", label=None, alpha=1.0
):
    """
    Draw joints and skeletal connections on a Matplotlib axis.
    joints: (J, 2) array or tensor of [x, y] coordinates.
    visibility: (J,) array or tensor of visibility flags (1 or True: draw, 0 or False: skip).
    """
    if torch.is_tensor(joints):
        joints = joints.cpu().numpy()

    if visibility is not None:
        if torch.is_tensor(visibility):
            visibility = visibility.cpu().numpy()
    else:
        # Default to all visible if not provided
        visibility = np.ones(len(joints))

    # Mask for valid joints
    is_visible = visibility.astype(bool)

    # Draw connections
    line_drawn = False
    for i, (j1, j2) in enumerate(LSP_SKELETON):
        if j1 < len(joints) and j2 < len(joints):
            # Only draw if both joints are visible
            if is_visible[j1] and is_visible[j2]:
                x = [joints[j1, 0], joints[j2, 0]]
                y = [joints[j1, 1], joints[j2, 1]]

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
            joints[is_visible, 0],
            joints[is_visible, 1],
            color=color,
            s=20,
            edgecolors="white",
            zorder=5,
            alpha=alpha,
        )


def compute_mpjpe(preds, gts, visibility=None):
    """
    Compute Mean Per Joint Position Error.
    preds: (B, J, 2)
    gts: (B, J, 2)
    visibility: (B, J) or (B, 3, J) - if (B, 3, J), index 2 is used as visibility.
    Returns: Average error per joint, and per-joint errors.
    """
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

    # Distance between preds and gts
    dist = np.sqrt(np.sum((preds - gts) ** 2, axis=-1))  # (B, J)

    # Apply visibility mask
    dist = dist * visibility

    per_joint_error = np.sum(dist, axis=0) / np.maximum(np.sum(visibility, axis=0), 1)
    mean_error = np.sum(dist) / np.maximum(np.sum(visibility), 1)

    return mean_error, per_joint_error


def compute_pck(preds, gts, visibility=None, threshold=15.0):
    """
    Compute Percentage of Correct Keypoints.
    threshold: Error threshold in pixels (after scaling to image size).
    """
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

    dist = np.sqrt(np.sum((preds - gts) ** 2, axis=-1))  # (B, J)

    correct = (dist <= threshold).astype(float) * visibility

    per_joint_pck = np.sum(correct, axis=0) / np.maximum(np.sum(visibility, axis=0), 1)
    mean_pck = np.sum(correct) / np.maximum(np.sum(visibility), 1)

    return mean_pck, per_joint_pck
