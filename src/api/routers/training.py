import json
import time
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional

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

RUNS_CACHE: Dict[str, Dict[str, Any]] = {}
RUN_DETAILS_CACHE: Dict[str, Dict[str, Any]] = {}

router = APIRouter()


@router.get("/training/status")
def get_training_status() -> Dict[str, Any]:
    return training_manager.get_status()


@router.post("/training/start")
def start_training(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    success, message = training_manager.start_training(config)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@router.post("/training/stop")
def stop_training() -> Dict[str, str]:
    success, message = training_manager.stop_training()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@router.get("/runs")
def list_runs() -> Dict[str, List[Dict[str, Any]]]:
    runs_dir = project_root / "results" / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    runs = []
    active_run_id = (
        training_manager.current_run_id if training_manager.is_running else None
    )

    try:
        run_paths = [x for x in runs_dir.iterdir() if x.is_dir()]
    except Exception:
        return {"runs": []}

    for run_path in run_paths:
        run_id = run_path.name

        # Check cache for non-active completed runs
        if run_id != active_run_id and run_id in RUNS_CACHE:
            cached_entry = RUNS_CACHE[run_id].copy()
            cached_entry["status"] = "completed"
            runs.append(cached_entry)
            continue

        try:
            stat_info = run_path.stat()
            run_info: Dict[str, Any] = {
                "id": run_id,
                "created_at": time.ctime(stat_info.st_ctime),
                "status": "active" if run_id == active_run_id else "completed",
                "st_ctime": stat_info.st_ctime,
            }
            eval_file = next(
                (
                    f
                    for f in [
                        run_path / "eval_results.json",
                        run_path / "evaluation.json",
                    ]
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

            # Only cache completed runs
            if run_id != active_run_id:
                RUNS_CACHE[run_id] = run_info

            runs.append(run_info)
        except Exception:
            pass

    # Sort runs by st_ctime descending
    runs.sort(key=lambda x: x.get("st_ctime", 0), reverse=True)

    # Return runs without st_ctime in the response
    return {"runs": [{k: v for k, v in r.items() if k != "st_ctime"} for r in runs]}


@router.get("/runs/{run_id}")
def get_run_details(run_id: str) -> Dict[str, Any]:
    active_run_id = (
        training_manager.current_run_id if training_manager.is_running else None
    )
    if run_id != active_run_id and run_id in RUN_DETAILS_CACHE:
        return RUN_DETAILS_CACHE[run_id]

    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    details: Dict[str, Any] = {"id": run_id}
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
        try:
            with open(eval_file, "r") as f:
                details["evaluation"] = format_evaluation_metrics(json.load(f))
        except Exception:
            pass

    # Cache details of completed runs
    if run_id != active_run_id:
        RUN_DETAILS_CACHE[run_id] = details

    return details


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> Dict[str, str]:
    run_path = project_root / "results" / "runs" / run_id
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        shutil.rmtree(run_path)
        # Invalidate caches
        RUNS_CACHE.pop(run_id, None)
        RUN_DETAILS_CACHE.pop(run_id, None)
        return {"message": f"Run {run_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
def evaluate_model(
    split: str = "val",
    checkpoint: Optional[str] = None,
    run_id: Optional[str] = None,
    force: bool = False,
    remote: bool = False,
) -> Dict[str, Any]:
    if split == "valid":
        split = "val"
    cache = load_evaluation_cache()
    checkpoint_path: Optional[Path] = None
    eval_config: Dict[str, Any] = {}
    checkpoint_key = checkpoint or (f"{run_id}_best" if run_id else "default")
    if checkpoint:
        checkpoint_path = project_root / "models" / "checkpoints" / Path(checkpoint)
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
        if inference_service._model is None:
            raise HTTPException(status_code=400, detail="Model not loaded")

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
def list_models() -> Dict[str, List[Dict[str, Any]]]:
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
