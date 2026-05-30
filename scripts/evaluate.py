"""
evaluate.py — Evaluate a trained pose estimation model on the validation set.

Metric: PCK@0.2 (Percentage of Correct Keypoints)
  A predicted joint is "correct" if its distance to ground truth is within
  20% of the torso diameter (right shoulder midpoint to left hip midpoint).

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
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Sampler

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, cast

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import VIPCupDataset, collate_skip_none  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    decode_heatmaps,
    LSP_JOINT_NAMES as JOINT_NAMES,
    draw_pose,
)

# LSP joint indices for torso diameter
R_SHOULDER = 8
L_HIP = 3


def compute_pck(
    pred_joints: torch.Tensor, gt_joints: torch.Tensor, threshold: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute PCK@threshold per joint.

    pred_joints: (N, J, 2) — predicted (x, y) in image space
    gt_joints:   (N, 3, J) — ground truth [x, y, visibility]
    threshold:   fraction of torso diameter to use as acceptance radius

    Visibility mask: vis <= 1 (includes both visible=0 and occluded=1).
    Joints with vis==2 (out-of-frame/unannotated) are excluded.

    Returns:
      per_joint_pck: (J) tensor, proportion correct per joint
      per_joint_count: (J) tensor, number of valid samples per joint
      mean_pck: float, mean PCK across all valid joints
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
    torso_dist_expanded = torso_dist.unsqueeze(1).expand(N, J, 1)  # (N, J, 1)
    torso_dist_final = torso_dist_expanded.clamp(min=1e-6)

    # Euclidean distance per joint
    dists = torch.norm(pred_joints - gt_xy, dim=-1)  # (N, J)
    correct = dists <= threshold * torso_dist_final.squeeze(-1)

    # Include visible (0) AND occluded (1) joints; exclude unannotated (2)
    valid = gt_vis <= 1  # (N, J)
    per_joint_correct = (correct & valid).sum(dim=0).float()
    per_joint_count = valid.sum(dim=0).float().clamp(min=1)
    per_joint_pck = per_joint_correct / per_joint_count

    valid_joints = valid.any(dim=0)
    mean_pck = float(per_joint_pck[valid_joints].mean().item())

    return per_joint_pck, per_joint_count, mean_pck


def load_run_config(checkpoint_path: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load the config embedded in the checkpoint (authoritative) or fall back
    to the run's config.json, then the global default.
    """
    if checkpoint_path.exists():
        state = cast(Dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        if isinstance(state, dict) and "config" in state:
            return cast(Dict[str, Any], state["config"]), state
    # Fallback: run-level config.json
    run_config = checkpoint_path.parent.parent / "config.json"
    if run_config.exists():
        with open(run_config) as f:
            cfg = cast(Dict[str, Any], json.load(f))
        return cfg, None
    from src.utils import load_config

    return load_config(), None


def calculate_skeleton_spread(joints: torch.Tensor) -> float:
    """
    joints: (J, 2)
    Returns: area of bounding box
    """
    min_coords = torch.min(joints, dim=0)[0]
    max_coords = torch.max(joints, dim=0)[0]
    diff = max_coords - min_coords
    return float((diff[0] * diff[1]).item())


def visualize_audit(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    image_size: Tuple[int, int],
    decode_method: str,
    save_path: Path,
) -> None:
    """
    Generates a set of sample inferences for visual verification.
    """
    samples_to_show = []
    # Pick one uncover, one cover1, one cover2
    covers_found = set()
    for i in range(len(dataset)):
        cover = dataset.samples[i]["cover"]
        if cover not in covers_found:
            samples_to_show.append(i)
            covers_found.add(cover)
        if len(samples_to_show) >= 4:
            break

    model.eval()
    fig, axes = plt.subplots(
        1, len(samples_to_show), figsize=(5 * len(samples_to_show), 5)
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes_list = [axes]
    else:
        axes_list = list(axes)

    with torch.no_grad():
        for i, idx in enumerate(samples_to_show):
            batch = dataset[idx]
            img = batch["image"].unsqueeze(0).to(device)
            gt_joints = batch["joints"]
            cover = dataset.samples[idx]["cover"]

            output = model(img)
            pred_joints = decode_heatmaps(
                output.cpu(), image_size, method=decode_method
            )[0]

            # Use original image for background if possible
            img_np = batch["image"][0].cpu().numpy()
            axes_list[i].imshow(img_np, cmap="gray")

            # Draw GT (Green) and Pred (Red)
            draw_pose(axes_list[i], gt_joints[:2, :].T, color="green", alpha=0.5, label="GT")
            draw_pose(axes_list[i], pred_joints, color="red", label="Pred")
            axes_list[i].set_title(f"Sample {idx} ({cover})")
            axes_list[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visual audit saved to {save_path}")


def evaluate(
    checkpoint_path: Union[str, Path],
    data_root: Optional[str] = None,
    batch_size: int = 16,
    pck_threshold: float = 0.2,
    save_json: Optional[str] = None,
    decode_method_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)

    # Load config from checkpoint (run-specific, authoritative)
    config, state = load_run_config(checkpoint_path)

    is_cyclegan = config.get("training_type") == "cyclegan" or config.get(
        "training", {}
    ).get("cyclegan", False)
    if is_cyclegan:
        rank = int(os.environ.get("RANK", -1))
        if rank <= 0:
            print(
                "Detected CycleGAN domain translation model. Running translation visual audit..."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from src.models.cyclegan.generator import GeneratorResNet

        input_shape = (3, 256, 256)
        model: torch.nn.Module = GeneratorResNet(input_shape, num_residual_blocks=6).to(device)

        if state is None:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = state.get("model_state_dict", state)
        # Strip metadata keys
        metadata_keys = ["decoding_config", "config", "best_optimized_pck"]
        filtered_state = {k: v for k, v in state_dict.items() if k not in metadata_keys}
        filtered_state = {
            k.replace("module.", ""): v for k, v in filtered_state.items()
        }

        model.load_state_dict(filtered_state)
        model.eval()

        dataset_cfg: Dict[str, Any] = config.get("dataset", {})
        data_root = data_root or dataset_cfg.get("root", "data/raw")

        val_dataset = VIPCupDataset(
            root=data_root or "data/raw",
            subjects=range(1, 31),
            covers=["uncover"],
            modalities=dataset_cfg.get("modalities", ["IR"]),
            split="train",
            in_channels=3,
            image_size=(256, 256),
        )

        audit_path = (
            checkpoint_path.parent.parent / f"visual_audit_{checkpoint_path.stem}.png"
        )

        num_samples = min(5, len(val_dataset))
        fig, axes_raw = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        axes = cast(np.ndarray, axes_raw)
        if num_samples == 1:
            axes = axes.reshape(1, 2)

        with torch.no_grad():
            for i in range(num_samples):
                sample = val_dataset[i]
                img_t = sample["image"]
                subj = sample["subject"]

                img_input = (img_t * 2) - 1.0
                img_input = img_input.unsqueeze(0).to(device)
                fake_B = model(img_input)
                fake_B = (fake_B.squeeze(0).cpu() + 1.0) / 2.0
                fake_B = torch.clamp(fake_B, 0, 1)

                axes[i][0].imshow(img_t.permute(1, 2, 0).numpy())
                axes[i][0].set_title(f"Original (Uncovered) - Subj {subj}")
                axes[i][0].axis("off")

                axes[i][1].imshow(fake_B.permute(1, 2, 0).numpy())
                axes[i][1].set_title("Generated (Covered)")
                axes[i][1].axis("off")

        plt.tight_layout()
        plt.savefig(audit_path)
        plt.close()

        if rank <= 0:
            print(f"CycleGAN Visual Audit saved to {audit_path}")
            metrics: Dict[str, Any] = {
                "model_type": "cyclegan",
                "run_id": config.get("run_id", "loop47_cyclegan"),
                "status": "success",
                "visual_audit": str(
                    audit_path.resolve().relative_to(project_root.resolve())
                ),
            }
            if save_json:
                save_path = Path(save_json)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                print(f"Results saved to {save_json}")
            return metrics
        return None

    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))
    data_root = data_root or dataset_cfg.get("root", "data/raw")
    s_val: List[int] = dataset_cfg.get("subjects_val", [81, 90])
    # Use decoding config from checkpoint if available, otherwise default
    decode_method = decode_method_override
    decode_temp = 10.0
    if state is not None and "decoding_config" in state:
        d_cfg = state["decoding_config"]
        if decode_method is None:
            decode_method = d_cfg.get("method", "argmax")
        decode_temp = d_cfg.get("temperature", 10.0)

    if decode_method is None:
        decode_method = "argmax"

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
        print(f"Decode method: {decode_method}")
        print(f"Image size: {image_size}")

    # Load model using run-specific config
    model = build_model(config).to(device)
    if rank <= 0:
        print(f"Loading: {checkpoint_path}")

    # Determine in_channels from model config
    model_cfg: Dict[str, Any] = config.get("model", {})
    model_name = str(model_cfg.get("name", "hrnet"))
    in_channels = int(model_cfg.get(model_name, {}).get("in_channels", 1))

    if state is None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # Strip metadata keys if present
    metadata_keys = ["decoding_config", "config", "best_optimized_pck"]
    filtered_state = {k: v for k, v in state_dict.items() if k not in metadata_keys}

    # Handle both DDP and non-DDP checkpoints
    if any(k.startswith("module.") for k in filtered_state.keys()):
        filtered_state = {
            k.replace("module.", ""): v for k, v in filtered_state.items()
        }

    # --- Compatibility Remapping ---
    remapped_state = {}
    model_keys = set(model.state_dict().keys())

    for k, v in filtered_state.items():
        new_k = k
        if "modules_list." in k:
            new_k = new_k.replace("modules_list.", "")
        if "fusion.fuse_layers" in new_k:
            new_k = new_k.replace("fusion.fuse_layers", "fuse_layers.layers")

        if new_k in model_keys:
            remapped_state[new_k] = v
        elif k in model_keys:
            remapped_state[k] = v
        else:
            if k.startswith("hrnet.") and k[6:] in model_keys:
                remapped_state[k[6:]] = v
            elif f"hrnet.{k}" in model_keys:
                remapped_state[f"hrnet.{k}"] = v
            else:
                remapped_state[k] = v

    load_res = model.load_state_dict(remapped_state, strict=False)
    missing = [k for k in load_res.missing_keys if "num_batches_tracked" not in k]
    if len(missing) > 0:
        if rank <= 0:
            print(f"Loaded with {len(missing)} missing keys (remapping applied).")
    else:
        if rank <= 0:
            print("Model loaded with 100% key parity.")

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    model.eval()

    # Setup Dataset — include uncover for sanity, plus cover1/2
    val_dataset = VIPCupDataset(
        root=data_root or "data/raw",
        subjects=range(s_val[0], s_val[1] + 1),
        modalities=dataset_cfg.get("modalities", ["IR"]),
        covers=["uncover", "cover1", "cover2"],
        split="valid",
        image_size=image_size,
        in_channels=in_channels,
    )

    if len(val_dataset) == 0:
        if rank <= 0:
            print(
                "WARNING: Validation set is empty. Check data_root and covers configuration."
            )
        return None

    val_sampler: Optional[Sampler] = (
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
        print(
            f"Validation samples: {len(val_dataset)} (subjects {s_val[0]}-{s_val[1]})"
        )

    per_joint_correct_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_count_total = torch.zeros(len(JOINT_NAMES), device=device)
    per_joint_error_total = torch.zeros(len(JOINT_NAMES), device=device)

    # Sanity check metrics
    total_spread_ratio = 0.0
    num_spread_samples = 0

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

            model_to_call = model.module if is_distributed else model
            if (
                hasattr(model_to_call, "forward")
                and "return_refined" in model_to_call.forward.__code__.co_varnames
            ):
                outputs, refined_coords = model(images, return_refined=True)
                preds = refined_coords.cpu()
                using_refined = True
            else:
                outputs = model(images)
                using_refined = False

            if targets is not None:
                # Use heatmaps for loss calculation
                total_loss += float(criterion(outputs, targets).item())
                num_batches += 1

            if not using_refined:
                raw_model = model.module if is_distributed else model
                if getattr(raw_model, "output_type", "heatmap") == "heatmap":
                    preds = decode_heatmaps(
                        outputs,
                        image_size,
                        method=decode_method,
                        temperature=decode_temp,
                    ).cpu()
                else:
                    preds = outputs.cpu()

            p_pck, p_count, _ = compute_pck(preds, joints, threshold=pck_threshold)

            # Euclidean distance per joint
            gt_xy = joints[:, :2, :].permute(0, 2, 1)  # (B, J, 2)
            gt_vis = joints[:, 2, :]  # (B, J)
            valid = (gt_vis <= 1).float()  # (B, J)
            dists = torch.norm(preds - gt_xy, dim=-1)  # (B, J)

            # Skeleton spread check
            for b in range(preds.shape[0]):
                pred_spread = calculate_skeleton_spread(preds[b])
                gt_spread = calculate_skeleton_spread(gt_xy[b])
                if gt_spread > 100:  # Ignore tiny skeletons (e.g. all joints occluded)
                    total_spread_ratio += pred_spread / gt_spread
                    num_spread_samples += 1

            per_joint_correct_total += (p_pck * p_count).to(device)
            per_joint_count_total += p_count.to(device)
            per_joint_error_total += (dists * valid).sum(dim=0).to(device)

    # Synchronize metrics across processes
    if is_distributed:
        dist.all_reduce(per_joint_correct_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_joint_count_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_joint_error_total, op=dist.ReduceOp.SUM)

        sync_spread = torch.tensor(
            [total_spread_ratio, float(num_spread_samples)], device=device
        )
        dist.all_reduce(sync_spread, op=dist.ReduceOp.SUM)
        total_spread_ratio = sync_spread[0].item()
        num_spread_samples = int(sync_spread[1].item())

        if num_batches > 0:
            loss_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())

    if rank <= 0:
        avg_loss = total_loss / max(num_batches, 1)
        mean_spread_ratio = total_spread_ratio / max(num_spread_samples, 1)

        per_joint_pck_np = (
            (per_joint_correct_total / per_joint_count_total.clamp(min=1)).cpu().numpy()
        )
        per_joint_error_np = (
            (per_joint_error_total / per_joint_count_total.clamp(min=1)).cpu().numpy()
        )
        mean_pck = per_joint_pck_np.mean()
        mean_mpjpe = per_joint_error_np.mean()

        print(f"\n=== Evaluation Results (MPJPE: {mean_mpjpe:.1f}px) ===")
        print(f"Skeleton Spread Ratio: {mean_spread_ratio:.2f} (Target: >0.7)")
        if mean_spread_ratio < 0.6:
            print("WARNING: Skeleton spread is low! Possible joint collapse detected.")
        print(f"=== PCK@{pck_threshold} Results ===")
        print(f"{'Joint':<15} {'PCK':>6} {'MPJPE':>8}")
        print("-" * 32)
        for name, pck, err in zip(JOINT_NAMES, per_joint_pck_np, per_joint_error_np):
            print(f"{name:<15} {pck * 100:>5.1f}% {err:>8.1f}px")
        print("-" * 32)
        print(f"{'Mean':<15} {mean_pck * 100:>5.1f}% {mean_mpjpe:>8.1f}px")

        # Visual Audit (MANDATORY)
        audit_path = (
            checkpoint_path.parent.parent / f"visual_audit_{checkpoint_path.stem}.png"
        )
        visualize_audit(
            model, val_dataset, device, image_size, decode_method, audit_path
        )

        metrics = {
            "pck": float(mean_pck),
            "mpjpe": float(mean_mpjpe),
            "loss": avg_loss,
            "skeleton_spread_ratio": mean_spread_ratio,
            "decode_method": decode_method,
            "image_size": list(image_size),
            "per_joint_pck": per_joint_pck_np.tolist(),
            "per_joint_mpjpe": per_joint_error_np.tolist(),
            "joint_names": JOINT_NAMES,
            "visual_audit": str(
                audit_path.resolve().relative_to(project_root.resolve())
            ),
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
    parser.add_argument(
        "--data_root", type=str, default=None, help="Override data root path."
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="PCK threshold as fraction of torso diameter (default: 0.2)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Path to save metrics JSON (e.g. results/runs/<run_id>/eval_results.json)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["argmax", "soft-argmax"],
        help="Override heatmap decoding method.",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path_str: str
    if args.run_id:
        if os.path.isabs(args.checkpoint):
            checkpoint_path_str = args.checkpoint
        else:
            checkpoint_path_str = (
                f"results/runs/{args.run_id}/checkpoints/{args.checkpoint}"
            )
        # Default save_json to the run directory
        if args.save_json is None:
            args.save_json = f"results/runs/{args.run_id}/eval_results.json"
    else:
        checkpoint_path_str = args.checkpoint

    if not os.path.exists(checkpoint_path_str):
        print(f"Checkpoint not found: {checkpoint_path_str}")
        sys.exit(1)

    evaluate(
        checkpoint_path_str,
        data_root=args.data_root,
        batch_size=args.batch_size,
        pck_threshold=args.threshold,
        save_json=args.save_json,
        decode_method_override=args.method,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
