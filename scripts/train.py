import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import load_config
from src.data.dataset import VIPCupDataset
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
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.get_argument = lambda x: None # dummy
    parser.add_argument("--data_root", type=str, default=dataset_cfg.get("root", "data/raw"))
    args, _ = parser.parse_known_args()
    
    data_root = args.data_root

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
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
    )

    # 5. Initialize Model
    model = get_pose_net(model_cfg).to(device)

    # 6. Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.0001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )
    criterion = nn.MSELoss()  # Heatmap loss

    # 7. Training Loop
    epochs = train_cfg.get("epochs", 10)
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for batch in pbar:
            images = batch["image"].to(device)
            targets = batch["target"]

            if targets is None:
                continue

            targets = targets.to(device)

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

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs(train_cfg.get("save_dir", "models/checkpoints"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(train_cfg.get("save_dir", "models/checkpoints"), f"hrnet_epoch_{epoch+1}.pth")
            )

    print("Training Complete!")


if __name__ == "__main__":
    train()
