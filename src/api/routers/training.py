import json
import time
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from torch.utils.data import DataLoader

from src.training.manager import training_manager
from src.data.dataset import collate_skip_none
from src.training.trainer import PoseTrainer
from src.api.inference import inference_service
from src.api.state import (
    project_root,
    dataset_container,
    format_evaluation_metrics,
    load_evaluation_cache,
    save_evaluation_cache,
)

router = APIRouter()


@router.get("/training/status")
async def get_training_status():
    return training_manager.get_status()


@router.post("/training/start")
async def start_training(config: dict = None):
    success, message = training_manager.start_training(config)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@router.post("/training/stop")
async def stop_training():
    success, message = training_manager.stop_training()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@router.get("/runs")
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


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    details = {"id": run_id}
    for f_name, key in [("config.json", "config"), ("history.json", "history")]:
        if (run_path / f_name).exists():
            with open(run_path / f_name, "r") as f:
                details[key] = json.load(f)

    # Add display metadata for frontend
    if "config" in details:
        from src.utils.config_manager import get_display_metadata_for_config

        details["display_metadata"] = get_display_metadata_for_config(details["config"])
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


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        shutil.rmtree(run_path)
        return {"message": f"Run {run_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
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


@router.get("/models")
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
