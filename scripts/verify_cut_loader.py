import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import VIPCupDataset
from src.data.augmentations import DataAugmenter

def main():
    config_path = Path("configs/loop59_cut_self_training.yaml")
    if not config_path.exists():
        print(f"Error: Config {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Instantiate DataAugmenter
    aug_config = config["training"]["augmentation"]
    # Override dataset_root in config
    aug_config["dataset_root"] = config["dataset"]["dataset_root"]
    
    print("Initializing DataAugmenter...")
    augmenter = DataAugmenter(config=aug_config, is_training=True)
    
    # 2. Instantiate Dataset
    dataset = VIPCupDataset(
        root=config["dataset"]["dataset_root"],
        subjects=config["dataset"]["subjects_train"],
        modalities=config["dataset"]["modalities"],
        covers=None, # Allow all covers (uncover, cover1, cover2)
        split="train",
        augmenter=augmenter,
        image_size=tuple(config["dataset"]["image_size"]),
        in_channels=3, # Loop 58 CUT uses 3 channels
    )
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Let's collect some samples to visualize:
    # 1. Uncovered sample where CUT is applied
    # 2. Uncovered sample where Advanced Cover is applied
    # 3. Uncovered sample where no cover is applied (clean)
    # 4. Covered sample (cover1 or cover2) to verify no double cover is applied
    
    samples_to_plot = []
    
    # We will search through the dataset
    uncover_samples = []
    covered_samples = []
    
    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Let's probe the dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cover_type = sample["cover"]
        
        # Check if we have image_source (returned when return_pair=True in train mode)
        if "image_source" not in sample:
            continue
            
        img = sample["image"]
        img_src = sample["image_source"]
        
        # Difference between original (pre-occlusion) and augmented
        diff = torch.abs(img - img_src).mean().item()
        
        if cover_type == "uncover":
            # This is an uncover sample. We want to see if CUT was applied or Adv Cover
            # Advanced cover typically uses histogram matching or FDA or copy-paste
            # Let's inspect the diff to see if an occlusion was applied.
            # If diff > 0.05, something was applied. Let's save a pool of samples.
            uncover_samples.append((idx, sample, diff))
        else:
            # Already covered sample
            covered_samples.append((idx, sample, diff))
            
        if len(uncover_samples) > 200 and len(covered_samples) > 200:
            break
            
    print(f"Found {len(uncover_samples)} uncover samples and {len(covered_samples)} covered samples for analysis.")
    
    # Sort uncover samples by diff to find examples of different augmentations
    uncover_samples.sort(key=lambda x: x[2], reverse=True)
    
    # Let's pick 3 uncover samples:
    # - High diff (likely CUT or Advanced Cover)
    # - Medium diff (likely CUT with alpha blend, or other)
    # - Low/Zero diff (likely No Cover, only minor intensity jitter)
    
    selected = []
    
    # 1. Uncovered with high modification
    for idx, sample, diff in uncover_samples:
        selected.append((idx, sample, "Uncovered (Modified)", diff))
        if len(selected) >= 4:
            break
            
    # 2. Covered samples
    for idx, sample, diff in covered_samples[:2]:
        selected.append((idx, sample, f"Covered ({sample['cover']})", diff))

    # Plot
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 4 * len(selected)))
    
    for i, (idx, sample, desc, diff) in enumerate(selected):
        img = sample["image"].permute(1, 2, 0).numpy()
        img_src = sample["image_source"].permute(1, 2, 0).numpy()
        
        # Clip to [0, 1]
        img = np.clip(img, 0, 1)
        img_src = np.clip(img_src, 0, 1)
        
        axes[i, 0].imshow(img_src)
        axes[i, 0].set_title(f"Source (Idx: {idx}, {desc})")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f"Augmented (Diff: {diff:.3f})")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    output_path = Path("results/verify_cut_loader.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
