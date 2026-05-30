import torch
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cyclegan.generator import GeneratorResNet
from src.data.dataset import VIPCupDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CycleGAN translation")
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID of the trained CycleGAN"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/raw", help="Path to dataset"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cyclegan_visual_audit.png",
        help="Output filename",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Generator
    run_dir = Path("results/runs") / args.run_id
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        # Try latest if best doesn't exist
        checkpoint_path = run_dir / "checkpoints" / "latest_model.pth"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    # We want G_AB (Uncovered -> Covered)
    # StandardTrainer saves model_state_dict which is G_AB
    gen_state = state.get("model_state_dict", state)
    # Remove 'module.' if present
    gen_state = {k.replace("module.", ""): v for k, v in gen_state.items()}

    input_shape = (3, 256, 256)
    generator = GeneratorResNet(input_shape, num_residual_blocks=6).to(device)
    generator.load_state_dict(gen_state)
    generator.eval()

    # 2. Load Dataset (Domain A: Uncovered)
    dataset = VIPCupDataset(
        args.data_root,
        subjects=range(1, 31),
        covers=["uncover"],
        modalities=["IR"],
        split="train",
        in_channels=3,
        image_size=(256, 256),
    )

    # 3. Translate and Plot
    fig, axes = plt.subplots(args.num_samples, 2, figsize=(10, 5 * args.num_samples))
    if args.num_samples == 1:
        axes = [axes]

    for i in range(args.num_samples):
        # Pick random sample
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        img_t = sample["image"]
        subject_id = sample["subject"]

        # Generator expects [-1, 1]
        img_input = (img_t * 2) - 1.0
        img_input = img_input.unsqueeze(0).to(device)

        with torch.no_grad():
            fake_B = generator(img_input)

        # Denormalize to [0, 1]
        fake_B = (fake_B.squeeze(0).cpu() + 1.0) / 2.0
        fake_B = torch.clamp(fake_B, 0, 1)

        # Plot Original (Uncovered)
        axes[i][0].imshow(img_t.permute(1, 2, 0).numpy())
        axes[i][0].set_title(f"Original (Uncovered) - Subj {subject_id}")
        axes[i][0].axis("off")

        # Plot Fake (Covered)
        axes[i][1].imshow(fake_B.permute(1, 2, 0).numpy())
        axes[i][1].set_title("Generated (Covered)")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
