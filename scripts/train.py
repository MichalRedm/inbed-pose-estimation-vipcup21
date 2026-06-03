"""
Unified training script for the SLP dataset.
Handles local/distributed execution, data loading, and trainer orchestration.
"""

import os
import torch
import argparse
import json
import torch.distributed as dist
import random
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.data.augmentations import DataAugmenter
from src.training.factory import create_trainer


def set_seed(seed: int = 42) -> None:
    """
    Seeds random number generators for reproducibility.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_cuda() -> None:
    """Prints CUDA availability and device name."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print(
                "WARNING: CUDA NOT AVAILABLE! Training on CPU will be extremely slow."
            )


def train() -> None:
    """
    Main training execution function.
    Parses arguments, initializes DDP (if configured), creates dataloaders,
    and runs the training loop via the designated Trainer.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 1. Parse Initial Config Path (to load before other overrides)
    parser = argparse.ArgumentParser(
        description="Unified Training Script", add_help=False
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args_config, _ = parser.parse_known_args()

    # 2. Load Configuration
    config = load_config(args_config.config)
    train_cfg: Dict[str, Any] = config.get("training", {})
    dataset_cfg: Dict[str, Any] = config.get("dataset", {})

    # 3. Parse CLI Overrides
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--data_root", type=str, default=dataset_cfg.get("root", "data/raw")
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lambda_anatomical", type=float, default=None)
    parser.add_argument("--lambda_coord", type=float, default=None)
    parser.add_argument("--lambda_coord_occluded", type=float, default=None)
    parser.add_argument("--sigma_start", type=float, default=None)
    parser.add_argument("--sigma_end", type=float, default=None)
    parser.add_argument(
        "--use_uncertainty_weighting",
        action="store_true",
        help="Enable uncertainty loss weighting",
    )
    parser.add_argument(
        "--uda", action="store_true", help="Enable Unsupervised Domain Adaptation"
    )
    parser.add_argument(
        "--cyclegan", action="store_true", help="Enable CycleGAN Domain Translation"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=None, help="Adversarial weight"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    args, _ = parser.parse_known_args()

    # 3. Apply Overrides to Config
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lambda_anatomical is not None:
        config["training"]["lambda_anatomical"] = args.lambda_anatomical
    if args.lambda_coord is not None:
        config["training"]["lambda_coord"] = args.lambda_coord
    if args.lambda_coord_occluded is not None:
        config["training"]["lambda_coord_occluded"] = args.lambda_coord_occluded
    if args.sigma_start is not None:
        config["training"]["sigma_start"] = args.sigma_start
    if args.sigma_end is not None:
        config["training"]["sigma_end"] = args.sigma_end
    if args.use_uncertainty_weighting:
        config["training"]["use_uncertainty_weighting"] = True

    if args.uda:
        config["training_type"] = "uda"
        if "uda" not in config:
            config["uda"] = {}
        config["uda"]["enabled"] = True
        if args.lambda_adv is not None:
            config["uda"]["lambda_adv"] = args.lambda_adv

    if args.cyclegan:
        config["training_type"] = "cyclegan"
        if "training" not in config:
            config["training"] = {}
        config["training"]["cyclegan"] = True

    # Handle Run ID and Logging
    run_root: Optional[Path] = None
    if args.run_id:
        run_root = Path.cwd() / "results" / "runs" / args.run_id

        os.makedirs(run_root / "checkpoints", exist_ok=True)
        if rank == 0:
            # Clean up any leftover .tmp files from previous crashed runs to save disk space
            for tmp_file in (run_root / "checkpoints").glob("*.tmp"):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
        config["training"]["save_dir"] = str(run_root)

        # Save config snapshot for reproducibility
        if rank == 0:
            with open(run_root / "config.json", "w") as f:
                json.dump(config, f, indent=4)
    else:
        pass

    # 4. Setup Device & Distributed
    set_seed(int(train_cfg.get("seed", 42)))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    has_cuda = torch.cuda.is_available()

    if is_distributed and has_cuda:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if has_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if rank == 0:
        mode = str(config.get("training_type", "standard")).upper()
        print(f"--- Starting {mode} Training ---", flush=True)
        print(
            f"Device: {device} (Distributed: {is_distributed}, World Size: {world_size})",
            flush=True,
        )
        sys.stdout.flush()

    # 5. Initialize Data
    s_train: List[int] = dataset_cfg.get("subjects_train", [1, 30])
    s_val: List[int] = dataset_cfg.get("subjects_val", [81, 90])
    augmenter = DataAugmenter(
        config["training"].get("augmentation", {}), dataset_root=args.data_root
    )

    # Determine in_channels from model config
    model_cfg: Dict[str, Any] = config.get("model", {})
    model_name = str(model_cfg.get("name", "hrnet"))
    in_channels = int(model_cfg.get(model_name, {}).get("in_channels", 1))

    if config.get("training", {}).get("cyclegan"):
        from src.data.dataset import PairedDataset

        # For CycleGAN, we want geometric augmentations (flip, rotate, scale)
        # and sensor noise/intensity jitter, but NO blanket simulation!
        gan_aug_cfg = config["training"].get("augmentation", {}).copy()
        gan_aug_cfg["occlusion_prob"] = 0.0  # Disable mathematical blanket simulation
        gan_aug_cfg["enabled"] = True
        gan_augmenter = DataAugmenter(gan_aug_cfg)

        # Domain A: Uncovered (Subjects 1-30)
        ds_A = VIPCupDataset(
            args.data_root,
            subjects=range(1, 31),
            covers=["uncover"],
            modalities=["IR"],
            split="train",
            augmenter=gan_augmenter,
            in_channels=3,
            return_joints=False,
        )
        # Domain B: Covered (Subjects 31-80)
        ds_B = VIPCupDataset(
            args.data_root,
            subjects=range(31, 81),
            covers=["cover1", "cover2"],
            modalities=["IR"],
            split="train",
            augmenter=gan_augmenter,
            in_channels=3,
            return_joints=False,
        )
        train_dataset: Union[VIPCupDataset, PairedDataset] = PairedDataset(ds_A, ds_B)

        # Validation for CycleGAN (using small subset of training subjects for monitoring)
        ds_A_val = VIPCupDataset(
            args.data_root,
            subjects=range(1, 6),
            covers=["uncover"],
            modalities=["IR"],
            split="train",
            in_channels=3,
            return_joints=False,
        )
        ds_B_val = VIPCupDataset(
            args.data_root,
            subjects=range(31, 36),
            covers=["cover1", "cover2"],
            modalities=["IR"],
            split="train",
            in_channels=3,
            return_joints=False,
        )
        val_dataset: Union[VIPCupDataset, PairedDataset] = PairedDataset(
            ds_A_val, ds_B_val
        )

        collate_fn: Optional[Any] = (
            None  # Standard collate is fine for PairedDataset returning tensors
        )
    else:
        train_dataset = VIPCupDataset(
            root=args.data_root,
            subjects=range(s_train[0], s_train[1] + 1),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            split="train",
            augmenter=augmenter,
            image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
            in_channels=in_channels,
        )
        val_dataset = VIPCupDataset(
            root=args.data_root,
            subjects=range(s_val[0], s_val[1] + 1),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=dataset_cfg.get("covers_val", dataset_cfg.get("covers", None)),
            split="valid",
            image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
            in_channels=in_channels,
        )
        collate_fn = collate_skip_none

    train_sampler: Optional[torch.utils.data.Sampler] = (
        torch.utils.data.DistributedSampler(train_dataset) if is_distributed else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4 if os.name != "nt" else 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 if os.name != "nt" else 0,
    )

    # 6. Initialize Trainer via Factory
    trainer, model = create_trainer(config, device, rank, world_size)

    # 7. Resume Logic (Robustly integrated)
    if args.resume:
        save_dir = config["training"].get("save_dir")
        if rank == 0:
            print(f"[RESUME] Checking for checkpoints in: {save_dir}", flush=True)

        if not save_dir:
            if rank == 0:
                print(
                    "[RESUME] ERROR: save_dir not found in config. Cannot resume.",
                    flush=True,
                )
        else:
            ckpt_root = Path(save_dir) / "checkpoints"
            if rank == 0:
                print(f"[RESUME] Checkpoint root: {ckpt_root.absolute()}", flush=True)
                if not ckpt_root.exists():
                    print(
                        f"[RESUME] ERROR: Checkpoint directory does not exist: {ckpt_root}",
                        flush=True,
                    )
                else:
                    print(
                        f"[RESUME] Checkpoint directory contents: {os.listdir(ckpt_root)}",
                        flush=True,
                    )

            ckpt_files = list(ckpt_root.glob("*.pth"))
            if rank == 0:
                print(f"[RESUME] Found {len(ckpt_files)} .pth files", flush=True)
                sys.stdout.flush()

            if ckpt_files:
                latest_model_path = ckpt_root / "latest_model.pth"
                if latest_model_path.exists():
                    latest_ckpt = latest_model_path
                else:
                    latest_ckpt = max(ckpt_files, key=os.path.getmtime)

                if rank == 0:
                    print(f"[RESUME] Loading checkpoint: {latest_ckpt}", flush=True)

                try:
                    state = torch.load(latest_ckpt, map_location=device)

                    # Get start_epoch from checkpoint state OR history (take max)
                    ckpt_epoch = int(state.get("epoch", 0))
                    hist_epoch = 0
                    history_path = Path(save_dir) / "history.json"
                    if history_path.exists():
                        try:
                            with open(history_path, "r") as f:
                                history_data = json.load(f)
                                hist_epoch = len(history_data)
                        except Exception as e:
                            if rank == 0:
                                print(
                                    f"[RESUME] Warning: could not read history.json: {e}",
                                    flush=True,
                                )

                    start_epoch = max(ckpt_epoch, hist_epoch)
                    trainer.start_epoch = start_epoch
                    if rank == 0:
                        print(
                            f"[RESUME] ckpt_epoch: {ckpt_epoch}, hist_epoch: {hist_epoch}",
                            flush=True,
                        )
                        print(
                            f"[RESUME] Resuming from global epoch {start_epoch + 1}",
                            flush=True,
                        )
                        sys.stdout.flush()

                    m_state: Dict[str, Any] = state.get("model_state_dict", state)
                    # Remove 'module.' prefix if it exists (saved from DDP)
                    m_state = {k.replace("module.", ""): v for k, v in m_state.items()}
                    model.load_state_dict(m_state)

                    # Use the new robust state restoration API
                    trainer.load_resume_state(state)
                except Exception as e:
                    if rank == 0:
                        print(
                            f"[RESUME] ERROR: Failed to load checkpoint: {e}",
                            flush=True,
                        )
            else:
                if rank == 0:
                    print(
                        "[RESUME] No checkpoint files found. Starting from scratch.",
                        flush=True,
                    )
                    sys.stdout.flush()

    # 8. Start Training
    trainer.fit(train_loader, val_loader)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
