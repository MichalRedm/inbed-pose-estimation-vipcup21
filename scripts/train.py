import os
import torch
import argparse
import glob
import re
import json
from tqdm import tqdm
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
from src.models.hrnet import get_pose_net


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


def train():
    # 1. Load Configuration
    config = load_config()
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {}).get("hrnet", {})
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
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in save_dir",
    )
    args, _ = parser.parse_known_args()

    data_root = args.data_root
    save_dir = train_cfg.get("save_dir", "models/checkpoints")
    start_epoch = 0

    # 2. Check for Remote Execution
    if config.get("remote", {}).get("use_remote", False):
        print("Triggering Remote Training...")
        # (This logic would involve uploading the project and running this script on the remote backend)
        # For now, we focus on the local implementation
        pass

    # 3. Setup Device & Distributed
    set_seed(train_cfg.get("seed", 42))

    # Determine if we are running in distributed mode (torchrun sets these env vars)
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        check_cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank <= 0:
        print(
            f"Using device: {device} (Distributed: {is_distributed}, World Size: {world_size})"
        )

    # 4. Initialize Data
    s_train = dataset_cfg.get("subjects_train", [1, 30])
    s_val = dataset_cfg.get("subjects_val", [81, 90])

    train_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(s_train[0], s_train[1] + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        split="train",
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
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else train_cfg.get("batch_size", 16)
    )

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
        sampler=val_sampler,
    )
    has_val = len(val_dataset) > 0
    if rank <= 0:
        if has_val:
            print(f"Validation samples: {len(val_dataset)}")
        else:
            print("No annotated validation samples found — skipping val loop.")

    # 5. Initialize Model
    model = get_pose_net(model_cfg).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # 6. Optimizer & Loss
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 0.0001)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )
    criterion = nn.MSELoss()  # Heatmap loss

    # 7. Resume Logic
    if args.resume:
        ckpt_files = glob.glob(os.path.join(save_dir, "*.pth"))
        if ckpt_files:
            # Sort by epoch number in filename: hrnet_epoch_10.pth
            def get_epoch(f):
                m = re.search(r"epoch_(\d+)", f)
                return int(m.group(1)) if m else 0

            latest_ckpt = max(ckpt_files, key=get_epoch)
            if rank <= 0:
                print(f"Resuming from checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_epoch = get_epoch(latest_ckpt)
        elif rank <= 0:
            print("No checkpoints found. Starting from scratch.")

    # 8. Training Loop
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 10)
    if rank <= 0:
        print(
            f"Starting training for {epochs} epochs (from epoch {start_epoch + 1})..."
        )

    for epoch in range(start_epoch, start_epoch + epochs):
        if rank <= 0:
            print(f"--- Epoch {epoch + 1}/{start_epoch + epochs} ---")
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        # Only rank 0 shows progress bar
        show_pbar = rank <= 0
        total_target_epochs = start_epoch + epochs
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{total_target_epochs}",
            disable=not show_pbar,
        )
        epoch_loss = 0

        for batch in pbar:
            if batch is None:  # entire batch had no annotations
                continue
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            outputs = model(images)

            # Heatmap MSE loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss across all local batches
        epoch_loss /= max(len(train_loader), 1)

        # Synchronize loss across all processes in distributed mode
        if is_distributed:
            loss_tensor = torch.tensor([epoch_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / world_size

        # 8b. Validation pass
        val_loss = None
        if has_val:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:  # entire batch had no annotations
                        continue
                    images = batch["image"].to(device)
                    targets = batch["target"].to(device)
                    outputs = model(images)
                    total_val_loss += criterion(outputs, targets).item()
                    val_batches += 1
            if val_batches > 0:
                val_loss = total_val_loss / val_batches

            if is_distributed and val_loss is not None:
                val_loss_tensor = torch.tensor([val_loss], device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item() / world_size

        # Save history and checkpoint (only rank 0)
        if rank <= 0:
            history_path = os.path.join(save_dir, "history.json")
            history = []
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    try:
                        history = json.load(f)
                    except json.JSONDecodeError:
                        history = []

            entry = {"epoch": epoch + 1, "train_loss": epoch_loss}
            if val_loss is not None:
                entry["val_loss"] = val_loss
            history.append(entry)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)

            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}: train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}: train_loss={epoch_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)
                # Unwrap model if DDP
                model_to_save = model.module if is_distributed else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(save_dir, f"hrnet_epoch_{epoch + 1}.pth"),
                )

    if is_distributed:
        dist.destroy_process_group()

    if rank <= 0:
        print("Training Complete!")


if __name__ == "__main__":
    train()
