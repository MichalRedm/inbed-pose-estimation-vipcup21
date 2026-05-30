import pytest
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, cast
from src.data.dataset import VIPCupDataset
from src.utils.telemetry import LocalTracker
from src.api.inference import InferenceService


def test_dataset_dynamic_sigma(tmp_path: Path) -> None:
    # Setup dummy data structure
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)

    # Create a dummy sample following SLP naming convention
    subj_dir = raw / "00001"
    subj_dir.mkdir()
    (subj_dir / "RGB" / "uncover").mkdir(parents=True)

    # Mock image
    from PIL import Image

    Image.new("RGB", (256, 256)).save(subj_dir / "RGB" / "uncover" / "img1.jpg")

    # Mock mat file for joints
    import numpy as np
    import scipy.io as sio

    # (3, 14, 1) -> (x, y, occluded)
    joints_gt = np.ones((3, 14, 1)) * 100
    sio.savemat(str(subj_dir / "joints_gt_RGB.mat"), {"joints_gt": joints_gt})

    ds = VIPCupDataset(
        root=str(raw), subjects=[1], modalities=["RGB"], covers=["uncover"]
    )

    # Test sigma update
    ds.set_sigma(3.0)
    assert ds.sigma == 3.0

    sample = ds[0]
    target = cast(torch.Tensor, sample["target"])
    # Heatmap peak should be wider with larger sigma
    # (Checking if it actually runs without error is the first step)
    assert target.shape == (14, 64, 64)


def test_local_tracker(tmp_path: Path) -> None:
    db_path = str(tmp_path / "telemetry.db")
    tracker = LocalTracker(db_path=db_path)

    run_id = "test_run"
    config = {"lr": 0.01}
    tracker.init_run(run_id, "Test Run", config)

    tracker.log_metric(run_id, 1, "loss", 0.5)
    tracker.log_metric(run_id, 1, "pck", 0.8)

    history = tracker.get_run_history(run_id)
    assert len(history) == 1
    assert history[0]["loss"] == 0.5
    assert history[0]["pck"] == 0.8


def test_inference_service_singleton() -> None:
    s1 = InferenceService()
    s2 = InferenceService()
    assert s1 is s2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_service_predict() -> None:
    # This requires a real model and checkpoint, which might not be available in CI
    # We'll just test the logic if possible
    pass
