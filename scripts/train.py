import os
import torch
import argparse
import re
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
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
from src.models import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for DDP weight initialization
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


from src.training.standard_trainer import StandardTrainer

def train():
    # 1. Load Configuration
    config = load_config()
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default=dataset_cfg.get("root", "data/raw")
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lambda_anatomical",
        type=float,
        default=None,
        help="Weight for anatomical constraint loss",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in save_dir",
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="Unique ID for this training run"
    )
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    run_root = None
    if args.run_id:
        run_root = Path(__file__).parent.parent / "results" / "runs" / args.run_id
        save_dir = str(run_root / "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        
        # Update config with CLI overrides for the snapshot
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        if args.lr is not None:
            config["training"]["lr"] = args.lr
        if args.lambda_anatomical is not None:
            config["training"]["lambda_anatomical"] = args.lambda_anatomical

        # Save config snapshot
        with open(run_root / "config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        # Update training save_dir to match run_id structure
        config["training"]["save_dir"] = str(run_root)
    else:
        save_dir = train_cfg.get("save_dir", "models/checkpoints")

    # 3. Setup Device & Distributed
    set_seed(train_cfg.get("seed", 42))

    # Determine if we are running in distributed mode
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
        print(f"Using device: {device} (Distributed: {is_distributed}, World Size: {world_size})")

    # 4. Initialize Data
    s_train = dataset_cfg.get("subjects_train", [1, 30])
    s_val = dataset_cfg.get("subjects_val", [81, 90])

    # Initialize Augmenter
    augmenter = DataAugmenter(config.get("training", {}).get("augmentation", {}))

    train_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(s_train[0], s_train[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        split="train",
        augmenter=augmenter,
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )
    val_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(s_val[0], s_val[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        covers=["cover1", "cover2"],
        split="valid",
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )

    num_workers = 4 if os.name != "nt" else 0
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    
    batch_size = config["training"].get("batch_size", 16)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # 5. Initialize Model
    model = build_model(config).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # 6. Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"].get("lr", 0.0001),
        weight_decay=config["training"].get("weight_decay", 0.0001),
    )
    criterion = nn.MSELoss()

    # 7. Resume Logic
    if args.resume:
        # Check in save_dir or run_root
        ckpt_root = Path(save_dir)
        ckpt_files = list(ckpt_root.glob("*.pth"))
        if not ckpt_files and run_root:
            ckpt_files = list((run_root / "checkpoints").glob("*.pth"))
            
        if ckpt_files:
            def get_epoch(f):
                m = re.search(r"epoch_(\d+)", f.name)
                return int(m.group(1)) if m else 0
            
            latest_ckpt = max(ckpt_files, key=get_epoch)
            if rank == 0:
                print(f"Resuming from checkpoint: {latest_ckpt}")
            
            ckpt = torch.load(latest_ckpt, map_location=device)
            m_state = ckpt.get("model_state_dict", ckpt)
            if is_distributed:
                model.module.load_state_dict(m_state)
            else:
                model.load_state_dict(m_state)
                
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif rank == 0:
            print("No checkpoints found. Starting from scratch.")

    # 8. Initialize and Run Trainer
    trainer = StandardTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    trainer.fit(train_loader, val_loader)

    if is_distributed:
        dist.destroy_process_group()

    if rank == 0:
        print("Training Complete!")


if __name__ == "__main__":
    train()
