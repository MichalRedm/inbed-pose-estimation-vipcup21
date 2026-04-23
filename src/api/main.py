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
from src.utils import load_config, decode_heatmaps, LSP_JOINT_NAMES  # noqa: E402
from src.models.hrnet import get_pose_net  # noqa: E402
from src.data.dataset import VIPCupDataset  # noqa: E402

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


# Dataset Endpoints
@app.get("/dataset/stats")
async def get_dataset_stats():
    stats = {}
    for split in ["train", "val"]:
        ds = dataset_container.get(split)
        if not ds:
            stats[split] = {"total": 0}
            continue

        modalities = {}
        covers = {}
        subjects = set()

        for sample in ds.samples:
            modalities[sample["modality"]] = modalities.get(sample["modality"], 0) + 1
            covers[sample["cover"]] = covers.get(sample["cover"], 0) + 1
            subjects.add(sample["subject"])

        stats[split] = {
            "total": len(ds),
            "modalities": modalities,
            "covers": covers,
            "subject_count": len(subjects),
        }

    return stats


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
        sample_info["id"] = i
        # Remove absolute path for security/privacy, just keep filename or relative path
        sample_info["filename"] = sample["image_path"].name
        del sample_info["image_path"]
        if "joints" in sample_info:
            # Convert numpy joints to list if present, but we might want to skip joints in list view
            del sample_info["joints"]
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
        "samples": filtered_samples[start:end]
    }


@app.get("/dataset/sample/{split}/{idx}")
async def get_sample_detail(split: str, idx: int):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = ds.samples[idx]
    # Re-load sample using __getitem__ to get processed data if needed, 
    # but here we just want the raw joints and metadata
    
    res = {
        "id": idx,
        "split": split,
        "subject": sample["subject"],
        "modality": sample["modality"],
        "cover": sample["cover"],
        "filename": sample["image_path"].name,
    }
    
    if sample["joints"] is not None:
        # joints is (3, 14)
        joints = sample["joints"]
        res["joints"] = []
        for i in range(joints.shape[1]):
            res["joints"].append({
                "name": LSP_JOINT_NAMES[i] if i < len(LSP_JOINT_NAMES) else f"Joint_{i}",
                "x": float(joints[0, i]),
                "y": float(joints[1, i]),
                "visible": int(joints[2, i]) == 0 # 0 is visible in SLP
            })
            
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

    # Initialize datasets
    try:
        root_path = project_root / dataset_cfg.get("root", "data/raw")
        print(f"Initializing datasets from root: {root_path}")
        dataset_container["train"] = VIPCupDataset(
            root=root_path,
            subjects=range(dataset_cfg.get("subjects_train", [1, 30])[0], dataset_cfg.get("subjects_train", [1, 30])[1] + 1),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="train"
        )
        dataset_container["val"] = VIPCupDataset(
            root=root_path,
            subjects=range(dataset_cfg.get("subjects_val", [81, 90])[0], dataset_cfg.get("subjects_val", [81, 90])[1] + 1),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="valid"
        )
        print(f"Datasets initialized. Train: {len(dataset_container['train'])} samples, Val: {len(dataset_container['val'])} samples")
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
