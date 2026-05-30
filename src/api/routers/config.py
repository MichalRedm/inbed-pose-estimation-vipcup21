import sys
import json
import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, cast

from src.api.state import project_root, GPUConfig
from src.utils import get_training_config, save_training_config

router = APIRouter()


@router.get("/config/gpu")
async def get_gpu_config() -> Dict[str, Any]:
    paths = [
        project_root / "gpu_connection.json",
        Path("gpu_connection.json"),
        Path(__file__).parent.parent.parent.parent / "gpu_connection.json",
    ]
    json_path = next((p for p in paths if p.exists()), None)
    if not json_path:
        return {}
    try:
        with open(json_path, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {str(e)}")


@router.post("/config/gpu")
async def save_gpu_config(config: GPUConfig) -> Dict[str, str]:
    json_path = project_root / "gpu_connection.json"
    try:
        with open(json_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        return {"message": "Configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.get("/config/training")
async def get_training_settings() -> Dict[str, Any]:
    try:
        return get_training_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@router.post("/config/training")
async def save_training_settings(config: Dict[str, Any]) -> Dict[str, str]:
    try:
        save_training_config(config)
        return {"message": "Training configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.post("/gpu/verify")
def verify_gpu() -> Dict[str, Any]:
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
