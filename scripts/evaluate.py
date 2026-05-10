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
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_config,
    decode_heatmaps,
    compute_mpjpe,
    LSP_JOINT_NAMES as JOINT_NAMES,
)
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.models import build_model

# Leeds Sports Pose joint indices (matches dataset README order)
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

    return per_joint_pck, per_joint_count, mean_pck


def evaluate(
    checkpoint_path, data_root, batch_size=16, pck_threshold=0.5, save_json=None
):
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

    # --- Setup Device & Distributed ---
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank != -1

    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank <= 0:
        print(
            f"Using device: {device} (Distributed: {is_distributed}, World Size: {world_size})"
        )

    # Load model using factory
    model = build_model(config).to(device)
    if rank <= 0:
        print(f"Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle both DDP and non-DDP checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

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
        if rank <= 0:
            print(
                "WARNING: Validation set is empty. Check data_root and covers configuration."
            )
        return

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
        collate_fn=collate_skip_none,
        sampler=val_sampler,
    )

    if rank <= 0:
        print(f"Validation samples: {len(val_dataset)}")

    per_joint_correct_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_count_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_error_total = torch.zeros(len(JOINT_NAMES), device=device)
    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            joints = batch["joints"]
            targets = batch["target"].to(device) if "target" in batch else None

            # Skip unannotated samples
            if joints is None:
                continue

            outputs = model(images)

            # Loss calculation if targets available
            if targets is not None:
                total_loss += criterion(outputs, targets).item()
                num_batches += 1

            if (
                model.module.output_type == "heatmap"
                if is_distributed
                else model.output_type == "heatmap"
            ):
                preds = decode_heatmaps(outputs.cpu(), image_size, method="soft-argmax")  # (B, J, 2)
            else:
                preds = outputs.cpu()

            # Compute PCK counts for this batch
            p_pck, p_count, _ = compute_pck(preds, joints, threshold=pck_threshold)

            # Compute MPJPE for this batch
            gt_xy = joints[:, :2, :].permute(0, 2, 1)  # (B, J, 2)
            _, p_error_arr = compute_mpjpe(preds, gt_xy, visibility=joints)

            # Convert to tensors and accumulate
            per_joint_correct_total += (p_pck * p_count).to(device)
            per_joint_count_total += p_count.to(device)

            p_error_tensor = torch.from_numpy(p_error_arr).float()
            per_joint_error_total += (p_error_tensor * p_count.cpu()).to(device)

    # Synchronize metrics across processes
    if is_distributed:
        dist.all_reduce(per_joint_correct_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_joint_count_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_joint_error_total, op=dist.ReduceOp.SUM)

        if num_batches > 0:
            loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())

    if rank <= 0:
        avg_loss = total_loss / max(num_batches, 1)
        per_joint_pck = (
            (per_joint_correct_total / per_joint_count_total.clamp(min=1)).cpu().numpy()
        )
        per_joint_error = (
            (per_joint_error_total / per_joint_count_total.clamp(min=1)).cpu().numpy()
        )
        mean_pck = per_joint_pck.mean()
        mean_mpjpe = per_joint_error.mean()

        print(
            f"\n=== Evaluation Results (Loss: {avg_loss:.4f}, MPJPE: {mean_mpjpe:.1f}) ==="
        )
        print(f"=== PCK@{pck_threshold} Results ===")
        print(f"{'Joint':<15} {'PCK':>6} {'MPJPE':>8}")
        print("-" * 32)
        for name, pck, err in zip(JOINT_NAMES, per_joint_pck, per_joint_error):
            print(f"{name:<15} {pck * 100:>5.1f}% {err:>8.1f}")
        print("-" * 32)
        print(f"{'Mean':<15} {mean_pck * 100:>5.1f}% {mean_mpjpe:>8.1f}")

        metrics = {
            "loss": avg_loss,
            "pck": float(mean_pck),
            "mpjpe": float(mean_mpjpe),
            "per_joint_pck": per_joint_pck.tolist(),
            "per_joint_error": per_joint_error.tolist(),
            "joint_names": JOINT_NAMES,
            "threshold": pck_threshold,
            "samples": len(val_dataset),
        }

        if save_json:
            import json

            save_path = Path(save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Results saved to {save_json}")

        return metrics

    return None


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
    parser.add_argument(
        "--save_json", type=str, default=None, help="Save metrics to JSON file"
    )
    parser.add_argument("--run_id", type=str, default=None, help="Run ID to evaluate")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if args.run_id:
        # If --checkpoint is provided, it's an absolute path.
        # If not, we construct it using run_id and checkpoint_name.
        if args.checkpoint == "models/checkpoints/hrnet_epoch_100.pth":
            checkpoint_name = os.environ.get("CHECKPOINT_NAME", "best_model.pth")
            checkpoint_path = (
                f"results/runs/{args.run_id}/checkpoints/{checkpoint_name}"
            )
        else:
            checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        # Try best_model.pth in models/checkpoints as fallback
        fallback = "models/checkpoints/best_model.pth"
        if os.path.exists(fallback):
            checkpoint_path = fallback
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

    evaluate(
        checkpoint_path,
        args.data_root,
        args.batch_size,
        args.threshold,
        save_json=args.save_json,
    )

    # Cleanup DDP
    if dist.is_initialized():
        dist.destroy_process_group()
