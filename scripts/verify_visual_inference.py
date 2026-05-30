import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Union, cast

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import build_model  # noqa: E402
from src.utils import decode_heatmaps, draw_pose  # noqa: E402


def verify_inference(run_id: str, image_path: str, output_dir: str = "results/debug_inference") -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = project_root / "results" / "runs" / run_id
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    with open(config_path, "r") as f:
        import json

        config = json.load(f)

    model = build_model(config).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    state = cast(Dict[str, Any], torch.load(checkpoint_path, map_location=device))
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # Load and preprocess image
    model_cfg: Dict[str, Any] = config.get("model", {})
    model_name = str(model_cfg.get("name", "hrnet"))
    in_channels = int(model_cfg.get(model_name, {}).get("in_channels", 1))

    image = Image.open(image_path)
    if in_channels == 3:
        image = image.convert("RGB")
    else:
        image = image.convert("L")
    model_image_size_list: List[int] = config.get("dataset", {}).get("image_size", [256, 256])
    model_image_size = (model_image_size_list[0], model_image_size_list[1])

    img_resized = image.resize(model_image_size)
    if in_channels == 3:
        img_tensor = (
            torch.from_numpy(np.array(img_resized))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
    else:
        img_tensor = (
            torch.from_numpy(np.array(img_resized)).unsqueeze(0).unsqueeze(0).float()
            / 255.0
        )
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        # Test both argmax and soft-argmax
        preds_argmax = decode_heatmaps(outputs.cpu(), model_image_size, method="argmax")
        preds_soft = decode_heatmaps(
            outputs.cpu(), model_image_size, method="soft-argmax", temperature=10.0
        )
        preds_soft_low = decode_heatmaps(
            outputs.cpu(), model_image_size, method="soft-argmax", temperature=1.0
        )

    # Plotting
    fig, axes_raw = plt.subplots(1, 3, figsize=(18, 6))
    axes = cast(np.ndarray, axes_raw)

    axes[0].imshow(img_resized, cmap="gray")
    draw_pose(axes[0], preds_argmax[0], color="red", label="Argmax")
    axes[0].set_title(f"Argmax (Run: {run_id})")

    axes[1].imshow(img_resized, cmap="gray")
    draw_pose(axes[1], preds_soft[0], color="blue", label="Soft-Argmax (T=10)")
    axes[1].set_title("Soft-Argmax T=10")

    axes[2].imshow(img_resized, cmap="gray")
    draw_pose(axes[2], preds_soft_low[0], color="green", label="Soft-Argmax (T=1)")
    axes[2].set_title("Soft-Argmax T=1")

    for ax in axes:
        ax.legend()

    save_path = os.path.join(output_dir, f"{run_id}_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved comparison to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test Loop 17 and Loop 9 if available
    img_path = r"data\raw\train\train\00001\IR\uncover\image_000001.png"

    verify_inference("loop17_uncertainty", img_path)
    # If loop9 exists
    if (project_root / "results" / "runs" / "loop9_anatomical_hinge").exists():
        verify_inference("loop9_anatomical_hinge", img_path)
