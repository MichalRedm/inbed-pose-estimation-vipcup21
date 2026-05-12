import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils import load_config, decode_heatmaps
from src.utils.pose import compute_pck, draw_pose
from src.models import build_model
from src.data.dataset import VIPCupDataset, collate_skip_none


def optimize_checkpoint(checkpoint_path, val_loader, device, num_samples=100):
    print(f"\nOptimizing: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Load state
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "config" in state:
        config = state["config"]
    else:
        config = load_config()

    # Build model
    model = build_model(config).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    image_size = tuple(config.get("dataset", {}).get("image_size", [256, 256]))

    # If it's not a heatmap model, we don't need to optimize decoding
    if not hasattr(model, "output_type") or model.output_type != "heatmap":
        print("Not a heatmap model, skipping.")
        return

    # Decoding strategies to test
    strategies = [
        {"method": "argmax", "temperature": 1.0},
        {"method": "soft-argmax", "temperature": 1.0},
        {"method": "soft-argmax", "temperature": 5.0},
        {"method": "soft-argmax", "temperature": 10.0},
        {"method": "soft-argmax", "temperature": 20.0},
        {"method": "soft-argmax", "temperature": 50.0},
        {"method": "soft-argmax", "temperature": 100.0},
    ]

    best_pck = -1.0
    best_strategy = strategies[0]

    # Collect predictions for a subset of samples
    all_outputs = []
    all_gts = []
    all_vis = []
    all_images = []

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images = batch["image"].to(device)
            outputs = model(images)

            all_outputs.append(outputs.cpu())
            all_gts.append(batch["joints"][:, :2, :].permute(0, 2, 1).numpy())
            all_vis.append((batch["joints"][:, 2, :] <= 1).numpy())
            all_images.append(images.cpu())

            count += images.size(0)
            if count >= num_samples:
                break

    all_outputs = torch.cat(all_outputs)
    all_gts = np.concatenate(all_gts)
    all_vis = np.concatenate(all_vis)

    # Test each strategy
    for strategy in strategies:
        preds = decode_heatmaps(
            all_outputs,
            image_size,
            method=strategy["method"],
            temperature=strategy["temperature"],
        ).numpy()

        pck, _ = compute_pck(preds, all_gts, visibility=all_vis)
        print(
            f"  {strategy['method']} (temp={strategy['temperature']}): PCK@0.5 = {pck:.4f}"
        )

        if pck > best_pck:
            best_pck = pck
            best_strategy = strategy

    print(f"  BEST: {best_strategy['method']} with PCK={best_pck:.4f}")

    # Update checkpoint
    if not isinstance(state, dict):
        state = {"model_state_dict": state}

    state["decoding_config"] = {
        "method": best_strategy["method"],
        "temperature": best_strategy["temperature"],
        "image_size": image_size,
    }
    state["best_optimized_pck"] = best_pck

    torch.save(state, checkpoint_path)
    print(f"  Saved optimized decoding config to {checkpoint_path}")

    # Visual Check
    visual_check_path = (
        checkpoint_path.parent / f"decoding_audit_{checkpoint_path.stem}.png"
    )

    # Take first sample
    img = all_images[0][0]  # (1, H, W)
    gt = all_gts[0]

    # Decode with best strategy
    best_preds = decode_heatmaps(
        all_outputs[0:1],
        image_size,
        method=best_strategy["method"],
        temperature=best_strategy["temperature"],
    )[0].numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(img[0], cmap="gray")
    draw_pose(plt.gca(), gt, color="green", label="Ground Truth")
    draw_pose(
        plt.gca(), best_preds, color="red", label=f"Pred ({best_strategy['method']})"
    )
    plt.title(
        f"Decoding Audit: {checkpoint_path.name}\nMethod: {best_strategy['method']} Temp: {best_strategy['temperature']} PCK: {best_pck:.4f}"
    )
    plt.legend()
    plt.axis("off")
    plt.savefig(visual_check_path)
    plt.close()
    print(f"  Visual check saved to {visual_check_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    # Load validation dataset
    dataset_cfg = config.get("dataset", {})
    root_path = Path(dataset_cfg.get("root", "data/raw"))

    val_ds = VIPCupDataset(
        root=root_path,
        subjects=range(81, 91),
        modalities=["IR"],
        covers=["uncover", "cover1", "cover2"],
        split="valid",
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_skip_none
    )

    # Find all checkpoints
    project_root = Path(__file__).parent.parent
    checkpoints = []

    # runs/
    runs_dir = project_root / "results" / "runs"
    if runs_dir.exists():
        for cp in runs_dir.glob("**/checkpoints/*.pth"):
            checkpoints.append(cp)

    # models/checkpoints/
    models_dir = project_root / "models" / "checkpoints"
    if models_dir.exists():
        for cp in models_dir.glob("*.pth"):
            checkpoints.append(cp)

    print(f"Found {len(checkpoints)} checkpoints to optimize.")

    for cp_path in checkpoints:
        try:
            optimize_checkpoint(cp_path, val_loader, device)
        except Exception as e:
            print(f"Error optimizing {cp_path}: {e}")


if __name__ == "__main__":
    main()
