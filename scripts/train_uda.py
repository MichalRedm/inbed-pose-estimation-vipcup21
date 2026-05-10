import os
import torch
import argparse
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
from src.models.discriminator import DomainDiscriminator
from src.training.uda_trainer import UDATrainer
from torch.nn.parallel import DistributedDataParallel as DDP

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    config = load_config()
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=dataset_cfg.get("root", "data/raw"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument("--lambda_adv", type=float, default=0.005, help="Weight for adversarial loss")
    parser.add_argument("--warmup_epochs", type=int, default=15, help="Epochs to reach full adversarial weight")
    args, _ = parser.parse_known_args()

    # Update config with CLI args for UDA
    if "uda" not in config: config["uda"] = {}
    config["uda"]["lambda_adv"] = args.lambda_adv
    config["uda"]["warmup_epochs"] = args.warmup_epochs
    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.run_id: config["training"]["save_dir"] = f"results/runs/{args.run_id}"

    set_seed(train_cfg.get("seed", 42))
    
    # Distributed setup
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_main = rank == 0
    
    # Data
    s_train = dataset_cfg.get("subjects_train", [1, 30])
    s_val = dataset_cfg.get("subjects_val", [81, 90])
    augmenter = DataAugmenter(train_cfg.get("augmentation", {}))

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

    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_skip_none, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_skip_none, num_workers=2, pin_memory=True
    )

    # Models
    model = build_model(config).to(device)
    discriminator = DomainDiscriminator(in_channels=480).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[local_rank])
    
    # Load Checkpoint
    checkpoint_path = args.checkpoint
    if args.resume and args.run_id:
        checkpoint_path = os.path.join(config["training"]["save_dir"], "checkpoints", "best_model.pth")

    if checkpoint_path and os.path.exists(checkpoint_path):
        if is_main: print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        # Handle cases where state might be the full checkpoint dict or just weights
        m_state = state.get("model_state_dict", state)
        if world_size > 1:
            model.module.load_state_dict(m_state, strict=False)
        else:
            model.load_state_dict(m_state, strict=False)
        
        # Load discriminator if available
        d_state = state.get("discriminator_state_dict")
        if d_state:
            if world_size > 1:
                discriminator.module.load_state_dict(d_state)
            else:
                discriminator.load_state_dict(d_state)

    # Optimizers
    lr = args.lr or train_cfg.get("lr", 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.MSELoss()

    # Initialize Trainer
    trainer = UDATrainer(
        model=model,
        discriminator=discriminator,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        criterion=criterion,
        config=config,
        device=device,
        rank=rank,
        world_size=world_size
    )

    if is_main:
        print("Starting UDA Refined Training (Loop 5)")
        print(f"Lambda Adv: {args.lambda_adv}, Warmup: {args.warmup_epochs}")

    trainer.fit(train_loader, val_loader)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
