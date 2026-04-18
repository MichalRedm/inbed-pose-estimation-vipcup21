import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import load_config
from src.data.dataset import VIPCupDataset
from src.models.hrnet import get_pose_net
from tqdm import tqdm


def train():
    # 1. Load Configuration
    config = load_config()
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {}).get("hrnet", {})
    dataset_cfg = config.get("dataset", {})

    # 2. Check for Remote Execution
    if config.get("remote", {}).get("use_remote", False):
        print("Triggering Remote Training...")
        # (This logic would involve uploading the project and running this script on the remote backend)
        # For now, we focus on the local implementation
        pass

    # 3. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Initialize Data
    train_dataset = VIPCupDataset(
        root=dataset_cfg.get("root", "data/raw"),
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
            joints = batch["joints"]

            if joints is None:
                continue

            joints = joints.to(device)

            # Forward pass
            outputs = model(images)

            # Simple joint coordinate regression loss for now
            # (In production, this should be heatmap-based)
            # Flatten outputs and calculate loss against target coordinates
            # This is a placeholder for the actual heatmap-to-coordinate logic
            loss = criterion(
                outputs.mean(dim=[2, 3]), joints[:, :2, :].mean(dim=2)
            )  # Placeholder

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    print("Training Complete!")


if __name__ == "__main__":
    train()
