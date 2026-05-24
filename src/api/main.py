import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
from pathlib import Path
import sys
import json
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import time
import base64
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any
import torchvision.transforms.v2 as v2

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.manager import training_manager  # noqa: E402
from src.utils import (  # noqa: E402
    get_training_config,
    save_training_config,
    LSP_JOINT_NAMES,
)
from src.data.dataset import VIPCupDataset, collate_skip_none  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from src.training.trainer import PoseTrainer  # noqa: E402
from src.api.inference import inference_service  # noqa: E402
from src.data.augmentations import (  # noqa: E402
    DataAugmenter,
    get_available_augmentations,
    apply_custom_augmentations,
)

app = FastAPI(
    title="In-Bed Pose Estimation API",
    description="API for predicting 14 human joints from in-bed images (RGB or IR).",
    version="1.0.0",
)

# Global storage for dataset objects
dataset_container = {}

# Serve static files from runs directory
runs_static_dir = project_root / "results" / "runs"
runs_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/runs", StaticFiles(directory=str(runs_static_dir)), name="runs")

EVALUATION_CACHE_FILE = project_root / "models" / "evaluation_cache.json"


class GPUConfig(BaseModel):
    type: str
    tunnel_hostname: str = ""
    host: str = ""
    ssh_user: str = "root"
    port: int = 22
    gpu: str = ""
    ssh_config_alias: str = ""
    proxy_command: str = ""


class AugmentationApplyRequest(BaseModel):
    split: str = "train"
    index: int
    modality: str = "IR"
    augmentations: list[dict] = []


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_evaluation_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to format raw evaluation metrics for dashboard display."""
    pj_pck = metrics.get("per_joint_pck")
    pj_error = metrics.get("per_joint_error", metrics.get("per_joint_mpjpe"))
    j_names = metrics.get("joint_names", LSP_JOINT_NAMES)

    if pj_pck is not None and pj_error is not None:
        metrics["per_joint_metrics"] = [
            {"name": name, "pck": float(pck), "error": float(error)}
            for name, pck, error in zip(j_names, pj_pck, pj_error)
        ]

        if "per_joint_pck" in metrics:
            del metrics["per_joint_pck"]
        if "per_joint_error" in metrics:
            del metrics["per_joint_error"]
        if "per_joint_mpjpe" in metrics:
            del metrics["per_joint_mpjpe"]

    for key in ["loss", "mpjpe", "pck"]:
        if key in metrics and metrics[key] is not None:
            metrics[key] = float(metrics[key])

    return metrics


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


# --- Basic & Config Endpoints ---


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


@app.get("/config/gpu")
async def get_gpu_config():
    paths = [
        project_root / "gpu_connection.json",
        Path("gpu_connection.json"),
        Path(__file__).parent.parent.parent / "gpu_connection.json",
    ]
    json_path = next((p for p in paths if p.exists()), None)
    if not json_path:
        return {}
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {str(e)}")


@app.post("/config/gpu")
async def save_gpu_config(config: GPUConfig):
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
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@app.post("/config/training")
async def save_training_settings(config: dict):
    try:
        save_training_config(config)
        return {"message": "Training configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@app.post("/gpu/verify")
def verify_gpu():
    json_path = project_root / "gpu_connection.json"
    if not json_path.exists():
        return {"success": False, "stdout": "", "stderr": "Config not found"}
    try:
        script_path = project_root / "scripts" / "verify_remote_gpu.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--json", str(json_path)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


# --- Training & Runs Endpoints ---


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
    active_run_id = (
        training_manager.current_run_id if training_manager.is_running else None
    )
    for run_path in sorted(
        runs_dir.iterdir(), key=lambda x: x.stat().st_ctime, reverse=True
    ):
        if not run_path.is_dir():
            continue
        run_info = {
            "id": run_path.name,
            "created_at": time.ctime(run_path.stat().st_ctime),
            "status": "active" if run_path.name == active_run_id else "completed",
        }
        eval_file = next(
            (
                f
                for f in [run_path / "eval_results.json", run_path / "evaluation.json"]
                if f.exists()
            ),
            None,
        )
        if eval_file:
            try:
                with open(eval_file, "r") as f:
                    eval_data = json.load(f)
                    run_info["eval_pck"] = eval_data.get("pck")
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
    for f_name, key in [("config.json", "config"), ("history.json", "history")]:
        if (run_path / f_name).exists():
            with open(run_path / f_name, "r") as f:
                details[key] = json.load(f)
    eval_file = next(
        (
            f
            for f in [run_path / "eval_results.json", run_path / "evaluation.json"]
            if f.exists()
        ),
        None,
    )
    if eval_file:
        with open(eval_file, "r") as f:
            details["evaluation"] = format_evaluation_metrics(json.load(f))
    return details


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        shutil.rmtree(run_path)
        return {"message": f"Run {run_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_model(
    split: str = "val",
    checkpoint: str = None,
    run_id: str = None,
    force: bool = False,
    remote: bool = False,
):
    if split == "valid":
        split = "val"
    cache = load_evaluation_cache()
    checkpoint_path = None
    eval_config = {}
    checkpoint_key = checkpoint or (f"{run_id}_best" if run_id else "default")
    if checkpoint:
        checkpoint_path = project_root / "models" / "checkpoints" / checkpoint
    elif run_id:
        checkpoint_path = (
            project_root
            / "results"
            / "runs"
            / run_id
            / "checkpoints"
            / "best_model.pth"
        )
        config_path = project_root / "results" / "runs" / run_id / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                eval_config = json.load(f)
    if checkpoint_path:
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        inference_service.load_model(str(checkpoint_path))
    if remote:
        if not run_id:
            raise HTTPException(
                status_code=400, detail="run_id required for remote eval"
            )
        if not training_manager._run_evaluation(is_remote=True, run_id=run_id):
            raise HTTPException(status_code=500, detail="Remote evaluation failed")
        eval_file = project_root / "results" / "runs" / run_id / "evaluation.json"
        with open(eval_file, "r") as f:
            metrics = json.load(f)
    else:
        ds = dataset_container.get(split) or (
            list(dataset_container.values())[0] if dataset_container else None
        )
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        loader = DataLoader(
            ds, batch_size=8, shuffle=False, collate_fn=collate_skip_none
        )
        trainer = PoseTrainer(
            inference_service._model,
            device=inference_service._device,
            config=eval_config,
        )
        metrics = trainer.evaluate(loader)
    metrics = format_evaluation_metrics(metrics)
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


# --- Dataset & Augmentation Endpoints ---


@app.get("/dataset/stats")
async def get_dataset_stats():
    summary = {
        "total": 0,
        "train": 0,
        "valid": 0,
        "modalities": ["IR", "RGB"],
        "covers": set(),
    }
    for dict_key, json_key in [("train", "train"), ("val", "valid")]:
        ds = dataset_container.get(dict_key)
        if not ds:
            continue
        summary[json_key] = len(ds)
        summary["total"] += len(ds)
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
        raise HTTPException(status_code=404, detail="Split not found")
    filtered = []
    for i, sample in enumerate(ds.samples):
        if (cover and sample["cover"] != cover) or (
            subject and sample["subject"] != subject
        ):
            continue
        mod = (
            "IR"
            if "IR" in sample["image_paths"]
            else list(sample["image_paths"].keys())[0]
        )
        filtered.append(
            {
                "index": i,
                "id": f"{split}_{i}",
                "subject": sample["subject"],
                "cover": sample["cover"],
                "modalities": list(sample["image_paths"].keys()),
                "has_joints": any(j is not None for j in sample["joints"].values()),
                "image_path": str(sample["image_paths"][mod]),
                "modality": mod,
            }
        )
    start, end = (page - 1) * limit, page * limit
    return {
        "total": len(filtered),
        "page": page,
        "limit": limit,
        "samples": filtered[start:end],
    }


@app.get("/dataset/sample/{split}/{idx}")
async def get_sample_detail(split: str, idx: int):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[idx]
    resolutions, joints_data = {}, {}
    for mod, path in sample["image_paths"].items():
        try:
            with Image.open(path) as img:
                resolutions[mod] = {"width": img.width, "height": img.height}
        except Exception:
            resolutions[mod] = {"width": 256, "height": 256}
        joints_data[mod] = (
            sample["joints"][mod][:2, :].T.tolist()
            if sample["joints"][mod] is not None
            else None
        )
    return {
        "id": idx,
        "split": split,
        "subject": sample["subject"],
        "cover": sample["cover"],
        "modalities": list(sample["image_paths"].keys()),
        "resolutions": resolutions,
        "joints_per_modality": joints_data,
    }


@app.get("/dataset/image/{split}/{idx}")
async def get_dataset_image(
    split: str, idx: int, modality: str = "IR", augment: bool = False
):
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[idx]
    image_path = sample["image_paths"].get(
        modality, list(sample["image_paths"].values())[0]
    )
    if not augment:
        return FileResponse(image_path)
    image = Image.open(image_path).convert("L" if modality == "IR" else "RGB")
    joints = sample["joints"].get(modality)
    train_config = (
        training_manager.config if hasattr(training_manager, "config") else {}
    )
    aug_cfg = train_config.get("training", {}).get("augmentation", {})
    augmenter = DataAugmenter(
        enabled=True,
        occlusion_prob=1.0,
        flip_prob=aug_cfg.get("flip_prob", 0.5),
        rotation_range=aug_cfg.get("rotation_range", [-30, 30]),
        scaling_range=aug_cfg.get("scaling_range", [0.8, 1.2]),
    )
    aug_img, _ = augmenter(image, joints, is_ir=(modality == "IR"))
    buf = io.BytesIO()
    aug_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/augmentations")
async def list_augmentations():
    return {"augmentations": get_available_augmentations()}


@app.post("/augmentations/apply")
async def apply_augmentations_endpoint(request: AugmentationApplyRequest):
    ds = dataset_container.get(request.split)
    if not ds or request.index >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[request.index]
    if request.modality not in sample["image_paths"]:
        raise HTTPException(status_code=404, detail="Modality not found")
    image = Image.open(sample["image_paths"][request.modality]).convert(
        "L" if request.modality == "IR" else "RGB"
    )
    aug_image, aug_joints = apply_custom_augmentations(
        image,
        sample["joints"].get(request.modality),
        request.augmentations,
        is_ir=(request.modality == "IR"),
    )
    if torch.is_tensor(aug_image):
        aug_image = v2.functional.to_pil_image(aug_image)
    buf = io.BytesIO()
    aug_image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    joints_list = []
    if aug_joints is not None:
        for i in range(aug_joints.shape[1]):
            joints_list.append(
                {
                    "name": LSP_JOINT_NAMES[i] if i < len(LSP_JOINT_NAMES) else f"J{i}",
                    "x": float(aug_joints[0, i]),
                    "y": float(aug_joints[1, i]),
                    "vis": float(aug_joints[2, i]),
                }
            )
    return {
        "image": f"data:image/png;base64,{img_str}",
        "joints": joints_list,
        "original_size": {"width": image.width, "height": image.height},
    }


# --- Inference Endpoints ---


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(None),
    run_id: str = Form(None),
    checkpoint: str = Form(None),
):
    try:
        checkpoint_path = None
        if run_id:
            checkpoint_path = (
                project_root
                / "results"
                / "runs"
                / run_id
                / "checkpoints"
                / (checkpoint or "best_model.pth")
            )
        elif model_name:
            checkpoint_path = project_root / "models" / "checkpoints" / model_name
        else:
            checkpoints = sorted(
                list((project_root / "models" / "checkpoints").glob("*.pth"))
            )
            if checkpoints:
                checkpoint_path = checkpoints[-1]
        if checkpoint_path and checkpoint_path.exists():
            inference_service.load_model(str(checkpoint_path))

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Determine channels from loaded model if possible
        in_channels = 1  # Default
        if inference_service._model:
            m = inference_service._model
            # Use standardized in_channels property if available
            if hasattr(m, "in_channels"):
                in_channels = m.in_channels
            # Legacy fallback
            elif hasattr(m, "model") and hasattr(m.model, "in_channels"):
                in_channels = m.model.in_channels

        image = image.convert("RGB" if in_channels == 3 else "L")
        orig_size = image.size
        model_size = (
            tuple(
                inference_service._config.get("dataset", {}).get(
                    "image_size", [256, 256]
                )
            )
            if inference_service._config
            else (256, 256)
        )
        img_resized = image.resize(model_size)
        img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
        if in_channels == 1:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        else:
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        preds = inference_service.predict(img_tensor)
        scale_x, scale_y = orig_size[0] / model_size[0], orig_size[1] / model_size[1]
        results = [
            {
                "joint": LSP_JOINT_NAMES[i] if i < len(LSP_JOINT_NAMES) else f"J{i}",
                "x": float(x) * scale_x,
                "y": float(y) * scale_y,
            }
            for i, (x, y) in enumerate(preds[0].cpu().numpy())
        ]
        return {
            "filename": file.filename,
            "original_size": {"width": orig_size[0], "height": orig_size[1]},
            "predictions": results,
        }
    except (UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"File must be an image: {str(e)}")
    except Exception as e:
        import traceback

        with open("api.log", "a") as log:
            log.write(f"  ERROR in predict: {str(e)}\n")
            log.write(traceback.format_exc() + "\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    train_config = (
        training_manager.config if hasattr(training_manager, "config") else {}
    )
    checkpoint_dir = Path(project_root) / "models" / "checkpoints"
    checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))
    if checkpoints:
        inference_service.load_model(str(checkpoints[-1]))
    try:
        dataset_cfg = train_config.get("dataset", {})
        root_path = project_root / dataset_cfg.get("root", "data/raw")
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
        dataset_container["val"] = VIPCupDataset(
            root=root_path,
            subjects=range(
                dataset_cfg.get("subjects_val", [81, 90])[0],
                dataset_cfg.get("subjects_val", [81, 90])[1] + 1,
            ),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="valid",
        )
    except Exception as e:
        print(f"Error initializing datasets: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "src")],
    )
