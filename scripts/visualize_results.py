import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils import load_config
from src.data.dataset import VIPCupDataset
from src.models.hrnet import get_pose_net


def visualize_samples(checkpoint_path, num_samples=3):
    # 1. Load Configuration
    config = load_config()
    model_cfg = config.get("model", {}).get("hrnet", {})
    dataset_cfg = config.get("dataset", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model
    model = get_pose_net(model_cfg).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Setup Dataset
    # We use a small subset for visualization
    dataset = VIPCupDataset(
        root=dataset_cfg.get("root", "data/raw"),
        subjects=[1],  # Use subject 1 as a sample
        modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
        image_size=tuple(dataset_cfg.get("image_size", [256, 256])),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 4. Generate Visualizations
    os.makedirs("results", exist_ok=True)

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break

            image = batch["image"].to(device)
            target = batch["target"].to(device)

            output = model(image)

            # Convert to numpy for plotting
            # Image is [C, H, W] -> [H, W, C]
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            # Normalize for display
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            # Take first few joint heatmaps for visualization
            target_np = torch.sum(target[0], dim=0).cpu().numpy()
            output_np = torch.sum(output[0], dim=0).cpu().numpy()

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(target_np, cmap="jet")
            axes[1].set_title("Target Heatmap")
            axes[1].axis("off")

            axes[2].imshow(output_np, cmap="jet")
            axes[2].set_title("Predicted Heatmap")
            axes[2].axis("off")

            save_path = f"results/sample_{count + 1}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to: {save_path}")

            count += 1


if __name__ == "__main__":
    import argparse

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
