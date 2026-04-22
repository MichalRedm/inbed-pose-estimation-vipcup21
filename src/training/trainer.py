import torch
from tqdm import tqdm


import os
import json


class PoseTrainer:
    """
    Handles the training and evaluation loop for pose estimation.
    Real implementation using Heatmap MSE loss.
    """

    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.epochs = config.get("training", {}).get("epochs", 10)
        self.save_dir = config.get("training", {}).get("save_dir", "models/checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        total_loss = 0

        for batch in pbar:
            if batch is None:
                continue

            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Heatmap MSE loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        return avg_loss

    def fit(self, train_loader, val_loader=None):
        print(f"Starting training on {self.device}...")
        history_path = os.path.join(self.save_dir, "history.json")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = None

            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

            # Update history
            self._update_history(history_path, epoch + 1, train_loss, val_loss)

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"hrnet_epoch_{epoch + 1}.pth"),
                )

    def _update_history(self, path, epoch, train_loss, val_loss):
        history = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                history = []

        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            entry["val_loss"] = val_loss
        history.append(entry)

        with open(path, "w") as f:
            json.dump(history, f, indent=4)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        batches = 0
        for batch in dataloader:
            if batch is None:
                continue
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = self.model(images)
            total_loss += self.criterion(outputs, targets).item()
            batches += 1
        return total_loss / max(batches, 1)
