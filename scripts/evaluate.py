"""
evaluate.py — Evaluate a trained pose estimation model on the validation set.

Metric: PCK@0.5 (Percentage of Correct Keypoints)
  A predicted joint is "correct" if its distance to ground truth is within
  50% of the torso diameter (right shoulder to left hip distance).

Usage:
  python scripts/evaluate.py --checkpoint models/checkpoints/hrnet_epoch_100.pth
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, decode_heatmaps
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.models import build_model

# Leeds Sports Pose joint indices (matches dataset README order)
JOINT_NAMES = [
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
# Torso diameter: distance between Right Shoulder (idx 8) and Left Hip (idx 3)
R_SHOULDER = 8
L_HIP = 3


def compute_pck(pred_joints, gt_joints, threshold=0.5):
    """
    Compute PCK@threshold per joint.

    pred_joints: (N, J, 2) — predicted (x, y) in image space
    gt_joints:   (N, 3, J) — ground truth [x, y, visibility]
    threshold:   fraction of torso diameter to use as acceptance radius

    Returns:
      per_joint_pck: (J,) array, proportion correct per joint
      mean_pck:      scalar, mean PCK across all visible joints
    """
    N, J, _ = pred_joints.shape

    # Ground truth coords: (N, 2, J) → (N, J, 2)
    gt_xy = gt_joints[:, :2, :].permute(0, 2, 1)  # (N, J, 2)
    gt_vis = gt_joints[:, 2, :]  # (N, J)  0=visible

    # Torso diameter per sample
    r_shoulder = gt_xy[:, R_SHOULDER, :]  # (N, 2)
    l_hip = gt_xy[:, L_HIP, :]  # (N, 2)
    torso_diam = torch.norm(r_shoulder - l_hip, dim=-1, keepdim=True)  # (N, 1)
    torso_diam = torso_diam.unsqueeze(1).expand(N, J, 1)  # (N, J, 1)

    # Euclidean distance per joint
    dist = torch.norm(pred_joints - gt_xy, dim=-1)  # (N, J)
    correct = dist < threshold * torso_diam.squeeze(-1)

    # Only count visible joints (vis == 0 in this dataset)
    visible = gt_vis == 0  # (N, J)
    per_joint_correct = (correct & visible).sum(dim=0).float()
    per_joint_count = visible.sum(dim=0).float().clamp(min=1)
    per_joint_pck = per_joint_correct / per_joint_count

    # Mean over all joints that have at least one visible sample
    valid_joints = visible.any(dim=0)
    mean_pck = per_joint_pck[valid_joints].mean().item()

    return per_joint_pck.numpy(), mean_pck


def evaluate(checkpoint_path, data_root, batch_size=16, pck_threshold=0.5):
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model using factory
    model = build_model(config).to(device)
    print(f"Loading: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Setup Dataset
    s_val = dataset_cfg.get("subjects_val", [81, 90])
    val_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(s_val[0], s_val[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        covers=["cover1", "cover2"],
        split="valid",
        image_size=image_size,
    )

    if len(val_dataset) == 0:
        print(
            "WARNING: Validation set is empty. Check data_root and covers configuration."
        )
        return

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_skip_none,
    )
    print(f"Validation samples: {len(val_dataset)}")

    all_preds = []
    all_gt = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            joints = batch["joints"]  # (B, 3, 14) tensor or None

            # Skip unannotated samples
            if joints is None:
                continue
            if isinstance(joints, torch.Tensor) and joints.shape[0] == 0:
                continue

            outputs = model(images)
            if model.output_type == "heatmap":
                preds = decode_heatmaps(outputs.cpu(), image_size)  # (B, J, 2)
            else:
                preds = outputs.cpu()

            all_preds.append(preds)
            all_gt.append(joints)

    if not all_preds:
        print("No annotated validation samples found.")
        return

    all_preds = torch.cat(all_preds, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    per_joint_pck, mean_pck = compute_pck(all_preds, all_gt, threshold=pck_threshold)

    print(f"\n=== PCK@{pck_threshold} Results ===")
    print(f"{'Joint':<15} {'PCK':>6}")
    print("-" * 22)
    for name, pck in zip(JOINT_NAMES, per_joint_pck):
        print(f"{name:<15} {pck * 100:>5.1f}%")
    print("-" * 22)
    print(f"{'Mean PCK':<15} {mean_pck * 100:>5.1f}%")

    return mean_pck


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/hrnet_epoch_100.pth",
    )
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="PCK threshold as fraction of torso diameter",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    evaluate(args.checkpoint, args.data_root, args.batch_size, args.threshold)
