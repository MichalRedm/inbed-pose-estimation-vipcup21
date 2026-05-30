import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, cast
from pydantic import BaseModel

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import LSP_JOINT_NAMES  # noqa: E402

# Global storage for dataset objects
dataset_container: Dict[str, Any] = {}

EVALUATION_CACHE_FILE: Path = project_root / "models" / "evaluation_cache.json"
runs_static_dir: Path = project_root / "results" / "runs"


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
    augmentations: List[Dict[str, Any]] = []


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


def load_evaluation_cache() -> Dict[str, Any]:
    if EVALUATION_CACHE_FILE.exists():
        try:
            with open(EVALUATION_CACHE_FILE, "r") as f:
                data = json.load(f)
                return cast(Dict[str, Any], data)
        except Exception:
            return {}
    return {}


def save_evaluation_cache(cache: Dict[str, Any]) -> None:
    EVALUATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)
