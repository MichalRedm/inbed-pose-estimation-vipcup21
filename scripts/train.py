import os
import torch
import argparse
import re
import json
import torch.distributed as dist
import random
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.data.augmentations import DataAugmenter
from src.training.factory import create_trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_cuda():
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print(
                "WARNING: CUDA NOT AVAILABLE! Training on CPU will be extremely slow."
            )


def train():
    # 1. Load Baseline Configuration
    config = load_config()
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    # 2. Parse CLI Arguments (Overrides)
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument(
        "--data_root", type=str, default=dataset_cfg.get("root", "data/raw")
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lambda_anatomical", type=float, default=None)
    parser.add_argument("--lambda_coord", type=float, default=None)
    parser.add_argument(
        "--uda", action="store_true", help="Enable Unsupervised Domain Adaptation"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=None, help="UDA Adversarial weight"
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

    if args.uda:
        config["training_type"] = "uda"
        if "uda" not in config:
            config["uda"] = {}
        config["uda"]["enabled"] = True
        if args.lambda_adv is not None:
            config["uda"]["lambda_adv"] = args.lambda_adv

    # Handle Run ID and Logging
    run_root = None
    if args.run_id:
        run_root = Path(__file__).parent.parent / "results" / "runs" / args.run_id
        os.makedirs(run_root / "checkpoints", exist_ok=True)
        config["training"]["save_dir"] = str(run_root)

        # Save config snapshot for reproducibility
        if int(os.environ.get("RANK", 0)) == 0:
            with open(run_root / "config.json", "w") as f:
                json.dump(config, f, indent=4)
    else:
        pass

    # 4. Setup Device & Distributed
    set_seed(train_cfg.get("seed", 42))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        check_cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        mode = config.get("training_type", "standard").upper()
        print(f"--- Starting {mode} Training ---")
        print(
            f"Device: {device} (Distributed: {is_distributed}, World Size: {world_size})"
        )

    # 5. Initialize Data
    s_train = dataset_cfg.get("subjects_train", [1, 30])
    s_val = dataset_cfg.get("subjects_val", [81, 90])
    augmenter = DataAugmenter(config["training"].get("augmentation", {}))

    train_dataset = VIPCupDataset(
        root=args.data_root,
        subjects=range(s_train[0], s_train[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        split="train",
        augmenter=augmenter,
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )
    val_dataset = VIPCupDataset(
        root=args.data_root,
        subjects=range(s_val[0], s_val[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        covers=["cover1", "cover2"],
        split="valid",
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )

    train_sampler = (
        torch.utils.data.DistributedSampler(train_dataset) if is_distributed else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_skip_none,
        num_workers=4 if os.name != "nt" else 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=4 if os.name != "nt" else 0,
    )

    # 6. Initialize Trainer via Factory
    trainer, model = create_trainer(config, device, rank, world_size)

    # 7. Resume Logic (Robustly integrated)
    if args.resume:
        ckpt_root = Path(config["training"]["save_dir"]) / "checkpoints"
        ckpt_files = list(ckpt_root.glob("*.pth"))
        if ckpt_files:

            def get_epoch(f):
                m = re.search(r"epoch_(\d+)", f.name)
                return int(m.group(1)) if m else 0

            latest_ckpt = max(ckpt_files, key=get_epoch)
            if rank == 0:
                print(f"Resuming from: {latest_ckpt}")

            state = torch.load(latest_ckpt, map_location=device)
            m_state = state.get("model_state_dict", state)
            if is_distributed:
                model.module.load_state_dict(m_state)
            else:
                model.load_state_dict(m_state)

            if "optimizer_state_dict" in state and hasattr(trainer, "optimizer"):
                trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
            if "optimizer_d_state_dict" in state and hasattr(trainer, "optimizer_d"):
                trainer.optimizer_d.load_state_dict(state["optimizer_d_state_dict"])
            if "discriminator_state_dict" in state and hasattr(
                trainer, "discriminator"
            ):
                trainer.discriminator.load_state_dict(state["discriminator_state_dict"])
            if "total_steps" in state and hasattr(trainer, "total_steps"):
                trainer.total_steps = state["total_steps"]

    # 8. Start Training
    trainer.fit(train_loader, val_loader)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
