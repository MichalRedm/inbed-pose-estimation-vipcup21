import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, cast
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, decode_heatmaps, draw_pose
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.models import build_model


def visualize_samples(checkpoint_path: str, num_samples: int = 3) -> None:
    # 1. Load Configuration
    config = load_config()
    dataset_cfg: Dict[str, Any] = config.get("dataset", {})
    image_size_list: List[int] = dataset_cfg.get("image_size", [256, 256])
    image_size = (image_size_list[0], image_size_list[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model using factory
    model = build_model(config).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_state = cast(Dict[str, Any], torch.load(checkpoint_path, map_location=device))
    if isinstance(checkpoint_state, dict) and "model_state_dict" in checkpoint_state:
        model.load_state_dict(checkpoint_state["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_state)
    model.eval()

    # 3. Setup Dataset
    s_train: List[int] = dataset_cfg.get("subjects_train", [1, 30])
    dataset = VIPCupDataset(
        root=str(dataset_cfg.get("root", "data/raw")),
        subjects=[s_train[0]],  # Use the first training subject as a sample
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        image_size=image_size,
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_skip_none
    )

    # 4. Generate Visualizations
    os.makedirs("results", exist_ok=True)

    count = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if count >= num_samples:
                break

            image = batch["image"].to(device)
            # gt_joints is (B, 3, 14) -> (3, 14) -> (14, 3) where [x, y, visibility]
            gt_joints = batch["joints"][0].cpu().numpy().T

            output = model(image)
            if getattr(model, "output_type", "heatmap") == "heatmap":
                pred_joints = decode_heatmaps(output, image_size)[0]  # (J, 2)
            else:
                pred_joints = output[0].cpu()  # (J, 2)

            # Convert image to numpy for plotting
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Left: Original Image
            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            # Right: Overlay
            axes[1].imshow(img_np)
            # SLP visibility logic: v=0 (visible), v=1 (occluded/blanket).
            # We want to show both as Ground Truth because they have valid locations.
            vis_mask = (gt_joints[:, 2] <= 1) & (gt_joints[:, 0] > 0)
            draw_pose(
                axes[1],
                gt_joints[:, :2],
                visibility=vis_mask,
                color="blue",
                label="Ground Truth",
                alpha=0.6,
            )
            draw_pose(axes[1], pred_joints, color="red", label="Predicted")
            axes[1].set_title(f"Pose Overlay (Sample {count + 1})")
            axes[1].legend(loc="upper right")
            axes[1].axis("off")

            plt.tight_layout()
            # Adjust top to prevent title cropping
            fig.subplots_adjust(top=0.9)
            save_path = f"results/sample_{count + 1}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved visualization to: {save_path}")

            count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/hrnet_epoch_100.pth"
    )
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        visualize_samples(args.checkpoint, args.num_samples)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
