from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

class TrainingStrategy:
    """Base class for training strategies."""
    def get_script_path(self, project_root: Path) -> Path:
        raise NotImplementedError

    def get_args(self, config: Dict[str, Any], run_id: Optional[str], is_resume: bool) -> List[str]:
        args = []
        if run_id:
            args.extend(["--run_id", run_id])
        if is_resume:
            args.append("--resume")
        return args

class StandardStrategy(TrainingStrategy):
    def get_script_path(self, project_root: Path) -> Path:
        return project_root / "scripts" / "train.py"

    def get_args(self, config: Dict[str, Any], run_id: Optional[str], is_resume: bool) -> List[str]:
        args = super().get_args(config, run_id, is_resume)
        uda_cfg = config.get("uda", {})
        if uda_cfg.get("enabled", False) or config.get("training_type") == "uda":
            args.append("--uda")
        return args

class CycleGANStrategy(TrainingStrategy):
    def get_script_path(self, project_root: Path) -> Path:
        return project_root / "scripts" / "train.py"

    def get_args(self, config: Dict[str, Any], run_id: Optional[str], is_resume: bool) -> List[str]:
        args = super().get_args(config, run_id, is_resume)
        args.append("--cyclegan")
        return args

def get_training_strategy(config: Dict[str, Any]) -> TrainingStrategy:
    """Factory to get the correct strategy based on config."""
    train_cfg = config.get("training", {})
    if train_cfg.get("cyclegan", False):
        return CycleGANStrategy()
    return StandardStrategy()
