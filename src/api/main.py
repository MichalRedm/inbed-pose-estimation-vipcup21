import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, decode_heatmaps, LSP_JOINT_NAMES  # noqa: E402
from src.models.hrnet import get_pose_net  # noqa: E402

app = FastAPI(
    title="In-Bed Pose Estimation API",
    description="API for predicting 14 human joints from in-bed images (RGB or IR).",
    version="1.0.0",
)

# Global model container
model_container = {}


@app.on_event("startup")
async def startup_event():
    config = load_config()
    model_cfg = config.get("model", {}).get("hrnet", {})
    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_pose_net(model_cfg).to(device)

    # Find latest checkpoint
    checkpoint_dir = Path(project_root) / "models" / "checkpoints"
    checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))

    if not checkpoints:
        print(
            f"WARNING: No checkpoints found in {checkpoint_dir}. Model will use random weights."
        )
    else:
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

    model.eval()

    model_container["model"] = model
    model_container["device"] = device
    model_container["image_size"] = image_size


@app.get("/")
async def root():
    return {"message": "In-Bed Pose Estimation API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        original_size = image.size
        image_resized = image.resize(model_container["image_size"])
        img_tensor = (
            torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        )
        img_tensor = img_tensor.unsqueeze(0).to(model_container["device"])

        # Inference
        with torch.no_grad():
            heatmaps = model_container["model"](img_tensor)
            preds = decode_heatmaps(heatmaps.cpu(), model_container["image_size"])

        # Rescale predictions to original image size
        # preds is (1, 14, 2)
        scale_x = original_size[0] / model_container["image_size"][0]
        scale_y = original_size[1] / model_container["image_size"][1]

        scaled_preds = preds[0].numpy()
        scaled_preds[:, 0] *= scale_x
        scaled_preds[:, 1] *= scale_y

        # Format response
        results = []
        for i, (x, y) in enumerate(scaled_preds):
            results.append(
                {
                    "joint": LSP_JOINT_NAMES[i]
                    if i < len(LSP_JOINT_NAMES)
                    else f"Joint_{i}",
                    "x": float(x),
                    "y": float(y),
                }
            )

        return {
            "filename": file.filename,
            "original_size": {"width": original_size[0], "height": original_size[1]},
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
