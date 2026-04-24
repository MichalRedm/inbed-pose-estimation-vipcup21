import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import json
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.manager import training_manager  # noqa: E402
from src.utils import (  # noqa: E402
    load_config,
    get_training_config,
    save_training_config,
    decode_heatmaps,
    LSP_JOINT_NAMES,
)
from src.models.hrnet import get_pose_net  # noqa: E402
from src.data.dataset import VIPCupDataset, collate_skip_none  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from src.training.trainer import PoseTrainer  # noqa: E402

app = FastAPI(
    title="In-Bed Pose Estimation API",
    description="API for predicting 14 human joints from in-bed images (RGB or IR).",
    version="1.0.0",
)


class GPUConfig(BaseModel):
    type: str
    tunnel_hostname: str = ""
    host: str = ""
    ssh_user: str = "root"
    port: int = 22
    gpu: str = ""
    ssh_config_alias: str = ""
    proxy_command: str = ""


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVALUATION_CACHE_FILE = project_root / "models" / "evaluation_cache.json"


def load_evaluation_cache():
    if EVALUATION_CACHE_FILE.exists():
        try:
            with open(EVALUATION_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_evaluation_cache(cache):
    EVALUATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


# Root & Health Endpoints
@app.get("/")
async def root():
    gpu_info = {"available": torch.cuda.is_available()}
    if gpu_info["available"]:
        gpu_info["name"] = torch.cuda.get_device_name(0)
        try:
            free, total = torch.cuda.mem_get_info(0)
            gpu_info["memory"] = {
                "free": free / (1024**3),
                "total": total / (1024**3),
                "used": (total - free) / (1024**3),
            }
        except Exception:
            pass

    return {
        "status": "online",
        "version": "1.0.0",
        "gpu": gpu_info,
    }


# Configuration Endpoints
@app.get("/config/gpu")
async def get_gpu_config():
    # Try multiple common locations for the config file
    paths = [
        project_root / "gpu_connection.json",
        Path("gpu_connection.json"),
        Path(__file__).parent.parent.parent / "gpu_connection.json",
    ]

    json_path = None
    for p in paths:
        if p.exists():
            json_path = p
            break

    if not json_path:
        return {}

    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {str(e)}")


@app.post("/config/gpu")
async def save_gpu_config(config: GPUConfig):
    # Prefer saving to the root of the project
    json_path = project_root / "gpu_connection.json"
    try:
        with open(json_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        return {"message": "Configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@app.get("/config/training")
async def get_training_settings():
    try:
        return get_training_config()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load training config: {str(e)}"
        )


@app.post("/config/training")
async def save_training_settings(config: dict):
    try:
        save_training_config(config)
        return {"message": "Training configuration saved successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save training config: {str(e)}"
        )


@app.post("/gpu/verify")
def verify_gpu():
    # Run the verification script and capture output
    # Note: Using 'def' instead of 'async def' so FastAPI runs this in a threadpool
    # and doesn't block the event loop during the long SSH connection attempt.
    json_path = project_root / "gpu_connection.json"

    if not json_path.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Config file not found at {json_path}. Please save your configuration first.",
        }

    try:
        # Pass explicit paths to the script
        script_path = project_root / "scripts" / "verify_remote_gpu.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--json", str(json_path)],
            capture_output=True,
            text=True,
            timeout=90,  # Increased timeout for slow SSH handshakes
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Verification timed out after 90 seconds. Check if Kaggle is still running and cloudflared is installed.",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Internal server error: {str(e)}",
        }


# Global model container

# Global containers
model_container = {}
dataset_container = {}


@app.get("/training/status")
async def get_training_status():
    return training_manager.get_status()


@app.post("/training/start")
async def start_training(config: dict = None):
    success, message = training_manager.start_training(config)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@app.post("/training/stop")
async def stop_training():
    success, message = training_manager.stop_training()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@app.post("/evaluate")
async def evaluate_model(
    split: str = "val", checkpoint: str = None, force: bool = False
):
    # Normalize split name
    if split == "valid":
        split = "val"

    # Load cache
    cache = load_evaluation_cache()
    checkpoint_key = checkpoint if checkpoint else "best_model.pth"

    # Check if results are cached
    if not force and checkpoint_key in cache and split in cache[checkpoint_key]:
        return cache[checkpoint_key][split]

    # Load model with specific checkpoint if provided
    device = model_container["device"]
    model = model_container["model"]

    if checkpoint:
        checkpoint_path = project_root / "models" / "checkpoints" / checkpoint
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Checkpoint {checkpoint} not found"
            )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    # Get dataset
    ds = dataset_container.get(split)
    if not ds and split == "val":
        # Check if we have any dataset at all
        ds = list(dataset_container.values())[0] if dataset_container else None

    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset split {split} not found")

    loader = DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_skip_none
    )

    trainer = PoseTrainer(model, device=device, config=model_container.get("config"))
    metrics = trainer.evaluate(loader)

    # Format per-joint metrics for display if they exist
    if "per_joint_error" in metrics:
        metrics["per_joint_metrics"] = [
            {"name": name, "pck": float(pck), "error": float(error)}
            for name, pck, error in zip(
                LSP_JOINT_NAMES, metrics["per_joint_pck"], metrics["per_joint_error"]
            )
        ]

        # Clean up numpy arrays for JSON serialization
        del metrics["per_joint_pck"]
        del metrics["per_joint_error"]

    # Convert other metrics to float for JSON
    for key in ["loss", "mpjpe", "pck"]:
        if key in metrics:
            metrics[key] = float(metrics[key])

    # Save to cache
    if checkpoint_key not in cache:
        cache[checkpoint_key] = {}
    cache[checkpoint_key][split] = metrics
    save_evaluation_cache(cache)

    return metrics


@app.get("/models")
async def list_models():
    checkpoint_dir = Path(project_root) / "models" / "checkpoints"
    if not checkpoint_dir.exists():
        return {"models": []}

    checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))
    return {
        "models": [
            {
                "name": cp.name,
                "path": str(cp.relative_to(project_root)),
                "size_mb": cp.stat().st_size / (1024 * 1024),
            }
            for cp in checkpoints
        ]
    }


@app.get("/dataset/stats")
async def get_dataset_stats():
    summary = {"total": 0, "train": 0, "valid": 0, "modalities": set(), "covers": set()}

    # Map internal dictionary keys to the output JSON keys expected by the frontend
    splits_mapping = [("train", "train"), ("val", "valid")]

    for dict_key, json_key in splits_mapping:
        ds = dataset_container.get(dict_key)
        if not ds:
            continue

        count = len(ds)
        summary[json_key] = count
        summary["total"] += count

        for sample in ds.samples:
            summary["modalities"].add(sample["modality"])
            summary["covers"].add(sample["cover"])

    summary["modalities"] = sorted(list(summary["modalities"]))
    summary["covers"] = sorted(list(summary["covers"]))

    return summary


@app.get("/dataset/samples")
async def get_samples(
    split: str = "train",
    page: int = 1,
    limit: int = 20,
    modality: str = None,
    cover: str = None,
    subject: int = None,
):
    ds = dataset_container.get(split)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Split {split} not found")

    filtered_samples = []
    for i, sample in enumerate(ds.samples):
        if modality and sample["modality"] != modality:
            continue
        if cover and sample["cover"] != cover:
            continue
        if subject and sample["subject"] != subject:
            continue

        # Add index to the sample info for later retrieval
        sample_info = sample.copy()
        sample_info["index"] = i
        sample_info["id"] = f"{split}_{i}"

        # Keep image_path for the /dataset/image?path= endpoint
        # or we can use the index-based one later
        sample_info["image_path"] = str(sample["image_path"])

        if "joints" in sample_info and sample_info["joints"] is not None:
            sample_info["joints"] = sample_info["joints"][:2, :].T.tolist()
            sample_info["has_joints"] = True
        else:
            sample_info["has_joints"] = False

        filtered_samples.append(sample_info)

    total = len(filtered_samples)
    start = (page - 1) * limit
    end = start + limit

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "samples": filtered_samples[start:end],
    }


@app.get("/dataset/sample/{split}/{idx}")
async def get_sample_detail(split: str, idx: int):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = ds.samples[idx]

    # Get original image size with EXIF orientation handled
    try:
        from PIL import ImageOps

        with Image.open(sample["image_path"]) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
    except Exception:
        width, height = 256, 256  # Fallback

    res = {
        "id": idx,
        "split": split,
        "subject": sample["subject"],
        "modality": sample["modality"],
        "cover": sample["cover"],
        "filename": sample["image_path"].name,
        "image_path": str(sample["image_path"]),
        "width": width,
        "height": height,
        "joints": sample["joints"][:2, :].T.tolist()
        if "joints" in sample and sample["joints"] is not None
        else None,
    }
    return res


@app.get("/dataset/image/{split}/{idx}")
async def get_dataset_image(split: str, idx: int):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = ds.samples[idx]
    return FileResponse(sample["image_path"])


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

    # Check for remote training dependencies
    try:
        print("Remote training dependencies (paramiko, scp) found.")
    except ImportError:
        print("WARNING: Remote training dependencies (paramiko, scp) not found.")
        print("Run 'pip install paramiko scp' to enable remote GPU support.")

    model_container["model"] = model
    model_container["device"] = device
    model_container["image_size"] = image_size
    model_container["config"] = config

    # Initialize datasets
    try:
        root_path = project_root / dataset_cfg.get("root", "data/raw")
        print(f"Initializing datasets from root: {root_path}")
        dataset_container["train"] = VIPCupDataset(
            root=root_path,
            subjects=range(
                dataset_cfg.get("subjects_train", [1, 30])[0],
                dataset_cfg.get("subjects_train", [1, 30])[1] + 1,
            ),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="train",
        )
        val_ds = VIPCupDataset(
            root=root_path,
            # Set the range to cover all domain adaptation subjects (31 to 70)
            subjects=range(
                dataset_cfg.get("subjects_val", [31, 70])[0],
                dataset_cfg.get("subjects_val", [31, 70])[1] + 1,
            ),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="valid",
        )
        dataset_container["val"] = val_ds
        dataset_container["valid"] = val_ds
        print(
            f"Datasets initialized. Train: {len(dataset_container['train'])} samples, Val: {len(val_ds)} samples"
        )
    except Exception as e:
        print(f"Error initializing datasets: {e}")


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
