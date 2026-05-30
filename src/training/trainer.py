import torch
from src.utils import decode_heatmaps, compute_mpjpe, compute_pck
from tqdm import tqdm


import os
import json


class PoseTrainer:
    """
    Handles the training and evaluation loop for pose estimation.
    Real implementation using Heatmap MSE loss.
    """

    def __init__(
        self, model, optimizer=None, criterion=None, device="cpu", config=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or torch.nn.MSELoss()
        self.device = device
        self.config = config or {}
        self.epochs = self.config.get("training", {}).get("epochs", 10)
        self.save_dir = self.config.get("training", {}).get(
            "save_dir", "models/checkpoints"
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        total_loss = 0

        for batch in pbar:
            if batch is None:
                continue

            images = batch["image"].to(self.device)

            image_size = self.config.get("dataset", {}).get("image_size", (256, 256))
            outputs = self.model(images)

            # Loss calculation
            if self.model.output_type == "heatmap":
                targets = batch["target"].to(self.device)
                loss = self.criterion(outputs, targets)
            else:
                targets = batch["joints"][:, :2, :].permute(0, 2, 1).to(self.device)
                # Normalize coordinates by image size to scale the training loss to the standard [0, 1] range
                h, w = image_size
                scale = torch.tensor([w, h], dtype=torch.float32, device=self.device)
                loss = self.criterion(outputs / scale, targets / scale)

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
        model_name = self.config.get("model", {}).get("name", "model")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = None

            if val_loader:
                val_result = self.evaluate(val_loader)
                val_loss = val_result["loss"]
                print(
                    f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"mpjpe={val_result.get('mpjpe', 0):.2f}"
                )
            else:
                print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

            # Update history
            self._update_history(history_path, epoch + 1, train_loss, val_loss)

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"{model_name}_epoch_{epoch + 1}.pth"),
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

        all_preds = []
        all_gts = []
        all_visibility = []

        image_size = self.config.get("dataset", {}).get("image_size", (256, 256))

        for batch in dataloader:
            if batch is None:
                continue
            images = batch["image"].to(self.device)
            joints = batch["joints"]  # (B, 3, 14)

            outputs = self.model(images)
            
            if self.model.output_type == "heatmap":
                targets = batch["target"].to(self.device)
                loss = self.criterion(outputs, targets)
            else:
                targets = batch["joints"][:, :2, :].permute(0, 2, 1).to(self.device)
                # Normalize coordinates by image size to scale the evaluation loss to the standard [0, 1] range
                h, w = image_size
                scale = torch.tensor([w, h], dtype=torch.float32, device=self.device)
                loss = self.criterion(outputs / scale, targets / scale)

            total_loss += loss.item()
            batches += 1

            # Decode predictions
            if self.model.output_type == "heatmap":
                preds = decode_heatmaps(
                    outputs, image_size, method="soft-argmax"
                ).cpu()  # (B, 14, 2)
            else:
                # If coordinates are predicted directly, they are already (B, 14, 2)
                preds = outputs.cpu()

            all_preds.append(preds)
            all_gts.append(
                joints[:, :2, :].permute(0, 2, 1)
            )  # (B, 3, 14) -> (B, 14, 2)
            all_visibility.append(joints)  # (B, 3, 14)

        avg_loss = total_loss / max(batches, 1)

        if not all_preds:
            return {"loss": avg_loss}

        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        all_visibility = torch.cat(all_visibility, dim=0)

        mpjpe, per_joint_error = compute_mpjpe(all_preds, all_gts, all_visibility)
        pck, per_joint_pck = compute_pck(all_preds, all_gts, all_visibility)

        return {
            "loss": avg_loss,
            "mpjpe": float(mpjpe),
            "pck": float(pck),
            "per_joint_error": per_joint_error.tolist(),
            "per_joint_pck": per_joint_pck.tolist(),
        }
