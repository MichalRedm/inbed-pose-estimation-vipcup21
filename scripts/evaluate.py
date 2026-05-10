"""
evaluate.py — Evaluate a trained pose estimation model on the validation set.

Metric: PCK@0.5 (Percentage of Correct Keypoints)
  A predicted joint is "correct" if its distance to ground truth is within
  50% of the torso diameter (right shoulder midpoint to left hip midpoint).

Usage:
  # Evaluate a specific run (recommended — uses the run's own config):
  python scripts/evaluate.py --run_id loop16_sigma_curriculum

  # Evaluate a specific checkpoint file:
  python scripts/evaluate.py --run_id loop16_sigma_curriculum --checkpoint epoch_10.pth

  # Override data root:
  python scripts/evaluate.py --run_id loop16_sigma_curriculum --data_root data/raw
"""

import os
import sys
import json
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    decode_heatmaps,
    LSP_JOINT_NAMES as JOINT_NAMES,
)
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.models import build_model

# LSP joint indices for torso diameter
R_SHOULDER = 8
L_HIP = 3


def compute_pck(pred_joints, gt_joints, threshold=0.5):
    """
    Compute PCK@threshold per joint.

    pred_joints: (N, J, 2) — predicted (x, y) in image space
    gt_joints:   (N, 3, J) — ground truth [x, y, visibility]
    threshold:   fraction of torso diameter to use as acceptance radius

    Visibility mask: vis <= 1 (includes both visible=0 and occluded=1).
    Joints with vis==2 (out-of-frame/unannotated) are excluded.

    Returns:
      per_joint_pck: (J,) tensor, proportion correct per joint
      per_joint_count: (J,) tensor, number of valid samples per joint
      mean_pck: scalar, mean PCK across all valid joints
    """
    N, J, _ = pred_joints.shape

    # Ground truth coords: (N, J, 2)
    gt_xy = gt_joints[:, :2, :].permute(0, 2, 1)
    gt_vis = gt_joints[:, 2, :]  # (N, J)

    # Torso distance per sample: distance between midpoint of shoulders and midpoint of hips
    # Indices: 8:RShoulder, 9:LShoulder, 2:RHip, 3:LHip
    shoulder_mid = (gt_xy[:, 8, :] + gt_xy[:, 9, :]) / 2.0
    hip_mid = (gt_xy[:, 2, :] + gt_xy[:, 3, :]) / 2.0
    torso_dist = torch.norm(shoulder_mid - hip_mid, dim=-1, keepdim=True)  # (N, 1)
    torso_dist = torso_dist.unsqueeze(1).expand(N, J, 1)                  # (N, J, 1)
    torso_dist = torso_dist.clamp(min=1e-6)

    # Euclidean distance per joint
    dists = torch.norm(pred_joints - gt_xy, dim=-1)  # (N, J)
    correct = dists <= threshold * torso_dist.squeeze(-1)

    # Include visible (0) AND occluded (1) joints; exclude unannotated (2)
    valid = gt_vis <= 1  # (N, J)
    per_joint_correct = (correct & valid).sum(dim=0).float()
    per_joint_count = valid.sum(dim=0).float().clamp(min=1)
    per_joint_pck = per_joint_correct / per_joint_count

    valid_joints = valid.any(dim=0)
    mean_pck = per_joint_pck[valid_joints].mean().item()

    return per_joint_pck, per_joint_count, mean_pck


def load_run_config(checkpoint_path: Path):
    """
    Load the config embedded in the checkpoint (authoritative) or fall back
    to the run's config.json, then the global default.
    """
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "config" in state:
            return state["config"], state
    # Fallback: run-level config.json
    run_config = checkpoint_path.parent.parent / "config.json"
    if run_config.exists():
        with open(run_config) as f:
            cfg = json.load(f)
        return cfg, None
    # Last resort: global default
    from src.utils import load_config
    return load_config(), None


def auto_decode_method(config: dict) -> str:
    """
    Select the correct heatmap decoding method based on the run's training config.
    - soft-argmax: for models trained with sigma curriculum (sigma_start != sigma_end)
    - argmax:      for models trained with standard fixed-sigma heatmap MSE
    """
    tc = config.get("training", {})
    sigma_start = tc.get("sigma_start", 2.0)
    sigma_end = tc.get("sigma_end", 2.0)
    if sigma_start != sigma_end:
        return "soft-argmax"
    return "argmax"


def evaluate(
    checkpoint_path,
    data_root=None,
    batch_size=16,
    pck_threshold=0.5,
    save_json=None,
):
    checkpoint_path = Path(checkpoint_path)

    # Load config from checkpoint (run-specific, authoritative)
    config, state = load_run_config(checkpoint_path)
    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))
    data_root = data_root or dataset_cfg.get("root", "data/raw")
    s_val = dataset_cfg.get("subjects_val", [81, 90])
    decode_method = auto_decode_method(config)

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
        print(f"Using device: {device} (Distributed: {is_distributed}, World Size: {world_size})")
        print(f"Decode method: {decode_method}")
        print(f"Image size: {image_size}")

    # Load model using run-specific config
    model = build_model(config).to(device)
    if rank <= 0:
        print(f"Loading: {checkpoint_path}")

    if state is None:
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # Handle both DDP and non-DDP checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    model.eval()

    # Setup Dataset — covered images only (task target domain)
    val_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(s_val[0], s_val[1] + 1),
        modalities=dataset_cfg.get("modalities", ["IR"]),
        covers=["cover1", "cover2"],
        split="valid",
        image_size=image_size,
    )

    if len(val_dataset) == 0:
        if rank <= 0:
            print("WARNING: Validation set is empty. Check data_root and covers configuration.")
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
        print(f"Validation samples: {len(val_dataset)} (subjects {s_val[0]}-{s_val[1]}, cover1+cover2)")

    per_joint_correct_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_count_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_error_total = torch.zeros(len(JOINT_NAMES), device=device)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            joints = batch["joints"]
            targets = batch["target"].to(device) if "target" in batch else None

            if joints is None:
                continue

            outputs = model(images)

            if targets is not None:
                total_loss += criterion(outputs, targets).item()
                num_batches += 1

            raw_model = model.module if is_distributed else model
            if raw_model.output_type == "heatmap":
                preds = decode_heatmaps(outputs.cpu(), image_size, method=decode_method)
            else:
                preds = outputs.cpu()

            p_pck, p_count, _ = compute_pck(preds, joints, threshold=pck_threshold)

            gt_xy = joints[:, :2, :].permute(0, 2, 1)   # (B, J, 2)
            gt_vis = joints[:, 2, :]                      # (B, J)
            valid = (gt_vis <= 1).float()                 # (B, J)
            dists = torch.norm(preds - gt_xy, dim=-1)      # (B, J)
            per_joint_error = (dists * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1)

            per_joint_correct_total += (p_pck * p_count).to(device)
            per_joint_count_total += p_count.to(device)
            per_joint_error_total += per_joint_error.to(device)

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

        print(f"\n=== Evaluation Results (Loss: {avg_loss:.4f}, MPJPE: {mean_mpjpe:.1f}px) ===")
        print(f"=== PCK@{pck_threshold} Results ===")
        print(f"{'Joint':<15} {'PCK':>6} {'MPJPE':>8}")
        print("-" * 32)
        for name, pck, err in zip(JOINT_NAMES, per_joint_pck, per_joint_error):
            print(f"{name:<15} {pck * 100:>5.1f}% {err:>8.1f}px")
        print("-" * 32)
        print(f"{'Mean':<15} {mean_pck * 100:>5.1f}% {mean_mpjpe:>8.1f}px")
        print(f"\nDecode method : {decode_method}")
        print(f"Image size    : {image_size}")
        print(f"Val subjects  : {s_val[0]}–{s_val[1]}, covers: cover1 + cover2")
        print("Visibility    : vis <= 1 (visible + occluded)")

        metrics = {
            "pck": float(mean_pck),
            "mpjpe": float(mean_mpjpe),
            "loss": avg_loss,
            "decode_method": decode_method,
            "image_size": list(image_size),
            "per_joint_pck": per_joint_pck.tolist(),
            "per_joint_mpjpe": per_joint_error.tolist(),
            "joint_names": JOINT_NAMES,
            "threshold": pck_threshold,
            "samples": len(val_dataset),
        }

        if save_json:
            save_path = Path(save_json)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Results saved to {save_json}")

        return metrics

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pose estimation model on the covered validation set."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID (e.g. loop16_sigma_curriculum). Config and checkpoint are loaded from results/runs/<run_id>/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename within the run's checkpoints/ dir (default: best_model.pth), "
             "or an absolute path to a .pth file.",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Override data root path.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="PCK threshold as fraction of torso diameter (default: 0.5)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Path to save metrics JSON (e.g. results/runs/<run_id>/eval_results.json)",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.run_id:
        if os.path.isabs(args.checkpoint):
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = f"results/runs/{args.run_id}/checkpoints/{args.checkpoint}"
        # Default save_json to the run directory
        if args.save_json is None:
            args.save_json = f"results/runs/{args.run_id}/eval_results.json"
    else:
        checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    evaluate(
        checkpoint_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        pck_threshold=args.threshold,
        save_json=args.save_json,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
