import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import json
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import time
import os
from fastapi.staticfiles import StaticFiles

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
from src.models import build_model  # noqa: E402
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from runs directory
runs_static_dir = project_root / "results" / "runs"
runs_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/runs", StaticFiles(directory=str(runs_static_dir)), name="runs")

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


# In-memory model cache: {model_key: {"model": model, "mtime": timestamp}}
model_container = {}


@app.get("/hello")
async def hello():
    return {"message": "Hello from API"}


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


@app.get("/runs")
async def list_runs():
    runs_dir = project_root / "results" / "runs"
    if not runs_dir.exists():
        return {"runs": []}

    runs = []
    # Get current active run from manager
    active_run_id = (
        training_manager.current_run_id if training_manager.is_running else None
    )

    for run_path in sorted(
        runs_dir.iterdir(), key=lambda x: x.stat().st_ctime, reverse=True
    ):
        if not run_path.is_dir():
            continue

        run_id = run_path.name
        run_info = {
            "id": run_id,
            "created_at": time.ctime(run_path.stat().st_ctime),
            "status": "active" if run_id == active_run_id else "completed",
            "has_config": (run_path / "config.json").exists(),
            "has_history": (run_path / "history.json").exists(),
            "has_eval": (run_path / "eval_results.json").exists()
            or (run_path / "evaluation.json").exists(),
            "has_audit": (run_path / "visual_audit_best_model.png").exists(),
        }

        # Load summary from history if available
        if run_info["has_history"]:
            try:
                with open(run_path / "history.json", "r") as f:
                    history = json.load(f)
                    if history:
                        run_info["epochs"] = len(history)
                        run_info["final_loss"] = history[-1].get("train_loss")
                        run_info["final_val_loss"] = history[-1].get("val_loss")
                        run_info["final_val_pck"] = history[-1].get("val_pck")
            except Exception:
                pass

        if run_info["has_eval"]:
            try:
                eval_file = run_path / "eval_results.json"
                if not eval_file.exists():
                    eval_file = run_path / "evaluation.json"
                with open(eval_file, "r") as f:
                    eval_data = json.load(f)
                    run_info["eval_pck"] = eval_data.get("pck")
                    run_info["eval_mpjpe"] = eval_data.get("mpjpe")
            except Exception:
                pass

        runs.append(run_info)

    return {"runs": runs}


@app.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    details = {"id": run_id}

    # Load config
    if (run_path / "config.json").exists():
        with open(run_path / "config.json", "r") as f:
            details["config"] = json.load(f)

    # Load history
    if (run_path / "history.json").exists():
        with open(run_path / "history.json", "r") as f:
            details["history"] = json.load(f)

    # Load evaluation results (check both standard filenames)
    eval_file = run_path / "eval_results.json"
    if not eval_file.exists():
        eval_file = run_path / "evaluation.json"

    if eval_file.exists():
        with open(eval_file, "r") as f:
            details["evaluation"] = json.load(f)

    # Visual audit path
    if (run_path / "visual_audit_best_model.png").exists():
        details["visual_audit_url"] = (
            f"/static/runs/{run_id}/visual_audit_best_model.png"
        )

    # List checkpoints
    ckpt_dir = run_path / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = sorted(list(ckpt_dir.glob("*.pth")))
        details["checkpoints"] = [
            {
                "name": cp.name,
                "size_mb": cp.stat().st_size / (1024 * 1024),
            }
            for cp in checkpoints
        ]

    return details


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        shutil.rmtree(run_path)
        return {"message": f"Run {run_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {str(e)}")


@app.post("/evaluate")
async def evaluate_model(
    split: str = "val",
    checkpoint: str = None,
    run_id: str = None,
    force: bool = False,
    remote: bool = False,
):
    # Normalize split name
    if split == "valid":
        split = "val"

    # Load cache
    cache = load_evaluation_cache()

    # Determine checkpoint path and config
    eval_config = model_container.get("config")
    if run_id:
        checkpoint_key = f"{run_id}:{checkpoint if checkpoint else 'best_model.pth'}"
        run_path = project_root / "results" / "runs" / run_id
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")

        checkpoint_name = checkpoint if checkpoint else "best_model.pth"
        checkpoint_path = run_path / "checkpoints" / checkpoint_name

        # Load run-specific config if available
        run_config_path = run_path / "config.json"
        if run_config_path.exists():
            with open(run_config_path, "r") as f:
                eval_config = json.load(f)
    else:
        checkpoint_key = checkpoint if checkpoint else "best_model.pth"
        checkpoint_path = project_root / "models" / "checkpoints" / checkpoint_key

    # Check if results are cached
    if not force and checkpoint_key in cache and split in cache[checkpoint_key]:
        return cache[checkpoint_key][split]

    # Load model with specific checkpoint if provided
    device = model_container["device"]
    model = model_container["model"]

    if checkpoint or run_id:
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Checkpoint not found at {checkpoint_path}"
            )
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    if remote:
        if not run_id:
            raise HTTPException(
                status_code=400, detail="run_id is required for remote evaluation"
            )

        # Trigger remote evaluation via manager helper
        # We run it synchronously here since the API is already blocking for local eval
        success = training_manager._run_evaluation(is_remote=True, run_id=run_id)

        if not success:
            raise HTTPException(status_code=500, detail="Remote evaluation failed")

        # Load the downloaded results
        eval_file = project_root / "results" / "runs" / run_id / "evaluation.json"
        if not eval_file.exists():
            raise HTTPException(
                status_code=500, detail="Evaluation results not found after remote run"
            )

        with open(eval_file, "r") as f:
            metrics = json.load(f)
    else:
        # Get dataset
        ds = dataset_container.get(split)
        if not ds and split == "val":
            # Check if we have any dataset at all
            ds = list(dataset_container.values())[0] if dataset_container else None

        if not ds:
            raise HTTPException(
                status_code=404, detail=f"Dataset split {split} not found"
            )

        loader = DataLoader(
            ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_skip_none
        )

        # Use the config from the run if available, otherwise global
        trainer = PoseTrainer(model, device=device, config=eval_config)
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
    summary = {
        "total": 0,
        "train": 0,
        "valid": 0,
        "modalities": ["IR", "RGB"],
        "covers": set(),
    }

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
            summary["covers"].add(sample["cover"])

    summary["covers"] = sorted(list(summary["covers"]))

    return summary


@app.get("/dataset/samples")
async def get_samples(
    split: str = "train",
    page: int = 1,
    limit: int = 20,
    cover: str = None,
    subject: int = None,
):
    ds = dataset_container.get(split)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Split {split} not found")

    filtered_samples = []
    for i, sample in enumerate(ds.samples):
        if cover and sample["cover"] != cover:
            continue
        if subject and sample["subject"] != subject:
            continue

        # Add index to the sample info for later retrieval
        sample_info = {
            "index": i,
            "id": f"{split}_{i}",
            "subject": sample["subject"],
            "cover": sample["cover"],
            "modalities": list(sample["image_paths"].keys()),
            "has_joints": any(j is not None for j in sample["joints"].values()),
        }

        # For the list view, we provide the IR path if available, else first modality
        default_mod = (
            "IR" if "IR" in sample["image_paths"] else sample_info["modalities"][0]
        )
        sample_info["image_path"] = str(sample["image_paths"][default_mod])
        sample_info["modality"] = default_mod

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

    # Get resolutions per modality
    resolutions = {}
    for mod, path in sample["image_paths"].items():
        try:
            with Image.open(path) as img:
                w, h = img.size
                resolutions[mod] = {"width": w, "height": h}
        except Exception:
            resolutions[mod] = {"width": 256, "height": 256}

    # Prepare joints - return a dictionary of joints per modality
    joints_data = {}
    for mod, joints in sample["joints"].items():
        if joints is not None:
            # joints is (3, 14) -> (x, y, visibility)
            joints_data[mod] = joints[:2, :].T.tolist()
        else:
            joints_data[mod] = None

    res = {
        "id": idx,
        "split": split,
        "subject": sample["subject"],
        "cover": sample["cover"],
        "modalities": list(sample["image_paths"].keys()),
        "resolutions": resolutions,
        "joints_per_modality": joints_data,
        "filenames": {
            mod: Path(path).name for mod, path in sample["image_paths"].items()
        },
    }
    return res


@app.get("/dataset/image/{split}/{idx}")
async def get_dataset_image(
    split: str, idx: int, modality: str = "IR", augment: bool = False
):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = ds.samples[idx]
    if modality not in sample["image_paths"]:
        # Fallback to first available if requested not found
        modality = list(sample["image_paths"].keys())[0]

    image_path = sample["image_paths"][modality]

    if not augment:
        return FileResponse(image_path)

    # Apply augmentation for preview
    image = Image.open(image_path)
    if modality == "IR":
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    joints = sample["joints"].get(modality)

    # Use the global training config for augmentation settings
    config = model_container.get("config", {})
    aug_cfg = config.get("training", {}).get("augmentation", {})

    from src.data.augmentations import DataAugmenter

    augmenter = DataAugmenter(
        enabled=True,
        occlusion_prob=1.0,  # Force occlusion for preview if it's the goal
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        rotation_range=aug_cfg.get("rotation_range", [-30, 30]),
        scaling_range=aug_cfg.get("scaling_range", [0.8, 1.2]),
    )

    augmented_image, _ = augmenter(image, joints, is_ir=(modality == "IR"))

    # Return as streaming response
    img_byte_arr = io.BytesIO()
    augmented_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.on_event("startup")
async def startup_event():
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    image_size = tuple(dataset_cfg.get("image_size", [256, 256]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model using factory
    model = build_model(config).to(device)

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
        state = torch.load(latest_checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model.eval()

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
            subjects=range(
                dataset_cfg.get("subjects_val", [81, 90])[0],
                dataset_cfg.get("subjects_val", [81, 90])[1] + 1,
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
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(None),
    run_id: str = Form(None),
    checkpoint: str = Form(None),
):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert(
            "L"
        )  # Convert to Grayscale (1-channel)

        # Preprocess
        original_size = image.size  # (W, H)

        # Determine checkpoint path
        checkpoint_path = None
        if run_id:
            checkpoint_name = checkpoint if checkpoint else "best_model.pth"
            checkpoint_path = (
                project_root
                / "results"
                / "runs"
                / run_id
                / "checkpoints"
                / checkpoint_name
            )
        elif model_name:
            checkpoint_path = project_root / "models" / "checkpoints" / model_name
        else:
            # Fallback to latest global checkpoint if nothing selected
            checkpoint_dir = Path(project_root) / "models" / "checkpoints"
            checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))
            if checkpoints:
                checkpoint_path = checkpoints[-1]

        # Determine image size and decoding method from run config
        model_image_size = (256, 256)
        # Default to argmax for Heatmap models (more robust peak detection)
        decode_method = "argmax"
        if checkpoint_path and (checkpoint_path.parent.parent / "config.json").exists():
            with open(checkpoint_path.parent.parent / "config.json", "r") as f:
                run_cfg = json.load(f)
                model_image_size = tuple(
                    run_cfg.get("dataset", {}).get("image_size", [256, 256])
                )
                # For future flexibility, we could add a "decode_method" field to config.json
                # For now, we prefer argmax for all HRNet heatmap models as soft-argmax
                # without high temperature causes joint clustering.
                if run_cfg.get("training", {}).get("force_soft_argmax", False):
                    decode_method = "soft-argmax"
                    print(f"[API] Forced soft-argmax decoding for {run_id}")
                else:
                    print(
                        f"[API] Using argmax decoding for {run_id} (Heatmap standard)"
                    )

        image_resized = image.resize(model_image_size)
        # (1, 1, H, W)
        img_tensor = (
            torch.from_numpy(np.array(image_resized)).unsqueeze(0).unsqueeze(0).float()
            / 255.0
        ).to(model_container["device"])

        # Check if file exists and get mtime
        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=404, detail=f"Checkpoint not found: {checkpoint_path}"
            )

        file_mtime = os.path.getmtime(checkpoint_path) if checkpoint_path else 0

        # Load model if not already loaded or if different run/checkpoint requested or if file is newer
        model_key = f"{model_name}_{run_id}_{checkpoint}"

        needs_load = False
        if model_key not in model_container:
            needs_load = True
        elif (
            isinstance(model_container[model_key], dict)
            and model_container[model_key].get("mtime", 0) < file_mtime
        ):
            print(f"[API] Checkpoint {checkpoint} updated on disk. Reloading...")
            needs_load = True

        # Inference
        device = model_container["device"]
        if needs_load and checkpoint_path:
            print(f"[API] Loading model: {model_name} from {checkpoint_path}")

            # Use run-specific config if available
            current_config = model_container["config"]
            if run_id:
                run_config_path = (
                    project_root / "results" / "runs" / run_id / "config.json"
                )
                if run_config_path.exists():
                    try:
                        with open(run_config_path, "r") as f:
                            current_config = json.load(f)
                        print(f"[API] Using run-specific config for {run_id}")
                    except Exception as e:
                        print(
                            f"[API] Warning: Failed to load run config, using global: {e}"
                        )

            # Build new model
            model = build_model(current_config).to(device)
            # Load state dict
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            model.eval()

            # Update container
            model_container[model_key] = {"model": model, "mtime": file_mtime}

        # Select active model from cache
        if model_key in model_container:
            active_model_entry = model_container[model_key]
            model = (
                active_model_entry["model"]
                if isinstance(active_model_entry, dict)
                else active_model_entry
            )
        elif "model" in model_container:
            model = model_container["model"]
        else:
            raise HTTPException(
                status_code=500,
                detail="No model available for inference. Check if checkpoints exist.",
            )

        with torch.no_grad():
            outputs = model(img_tensor)
            if model.output_type == "heatmap":
                preds = decode_heatmaps(
                    outputs.cpu(), model_image_size, method=decode_method
                )
            else:
                # Direct coordinates (1, 14, 2)
                preds = outputs.cpu()

        # Rescale predictions to original image size
        # preds is (1, 14, 2) in model_image_size space
        scale_x = original_size[0] / model_image_size[0]
        scale_y = original_size[1] / model_image_size[1]

        scaled_preds = preds[0].cpu().numpy().copy()
        for i in range(len(scaled_preds)):
            scaled_preds[i, 0] = float(scaled_preds[i, 0]) * scale_x
            scaled_preds[i, 1] = float(scaled_preds[i, 1]) * scale_y

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
        with open("api.log", "a") as log:
            log.write(f"  ERROR in predict: {str(e)}\n")
            import traceback

            log.write(traceback.format_exc() + "\n")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
