import os
import torch
import argparse
import glob
import re
import json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.models.hrnet import get_pose_net


def check_cuda():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA NOT AVAILABLE! Training on CPU will be extremely slow.")


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

    # 3. Setup Device
    check_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Initialize Data
    train_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(1, dataset_cfg.get("num_subjects_train", 30) + 1),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        split="train",
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )
    val_dataset = VIPCupDataset(
        root=data_root,
        subjects=range(71, 81),
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        covers=["cover1", "cover2"],
        split="valid",
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )

    num_workers = 4 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
    )
    has_val = len(val_dataset) > 0
    if has_val:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("No annotated validation samples found — skipping val loop.")

    # 5. Initialize Model
    model = get_pose_net(model_cfg).to(device)

    # 6. Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.0001),
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
            print(f"Resuming from checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_epoch = get_epoch(latest_ckpt)
        else:
            print("No checkpoints found. Starting from scratch.")

    # 8. Training Loop
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 10)
    print(f"Starting training from epoch {start_epoch + 1} to {epochs}...")

    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
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
            pbar.set_postfix(loss=loss.item())

        epoch_loss /= max(len(train_loader), 1)

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

        # Save history
        history_path = os.path.join(save_dir, "history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)

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
            os.makedirs(train_cfg.get("save_dir", "models/checkpoints"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    train_cfg.get("save_dir", "models/checkpoints"),
                    f"hrnet_epoch_{epoch + 1}.pth",
                ),
            )

    print("Training Complete!")


if __name__ == "__main__":
    train()
