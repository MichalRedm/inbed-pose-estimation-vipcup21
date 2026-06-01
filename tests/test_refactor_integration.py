import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import cast
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


def test_uda_lightning_module_compilation() -> None:
    from src.training.lightning_module import UDALightningModule
    from src.models.discriminator import DomainDiscriminator
    from typing import Any

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(
            self, x: torch.Tensor, return_features: bool = False, **kwargs: Any
        ) -> Any:
            features = torch.randn(x.size(0), 480, 8, 8)
            outputs = torch.randn(x.size(0), 14, 64, 64)
            if return_features:
                return outputs, features
            return outputs

    model = MockModel()
    discriminator = DomainDiscriminator(in_channels=480)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    config = {
        "uda": {"enabled": True, "lambda_adv": 0.001, "warmup_epochs": 10},
        "training": {"lr": 0.001, "weight_decay": 0.0001},
    }

    pl_module = UDALightningModule(
        model=model,
        discriminator=discriminator,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        criterion=criterion,
        config=config,
    )

    assert pl_module is not None
    assert pl_module.automatic_optimization is False

    # Mock batch
    batch = {
        "image": torch.randn(2, 1, 256, 256),
        "image_source": torch.randn(2, 1, 256, 256),
        "target": torch.randn(2, 14, 64, 64),
    }

    # Run one step
    pl_module.training_step(batch, 0)
    assert "loss" in pl_module.last_step_metrics
    assert "adv_loss" in pl_module.last_step_metrics


def test_cyclegan_lightning_module_compilation() -> None:
    from src.training.lightning_module import CycleGANLightningModule
    from src.models.cyclegan import GeneratorResNet, Discriminator

    G_AB = GeneratorResNet((3, 64, 64), num_residual_blocks=1, pretrained=False)
    G_BA = GeneratorResNet((3, 64, 64), num_residual_blocks=1, pretrained=False)
    D_A = Discriminator((3, 64, 64))
    D_B = Discriminator((3, 64, 64))

    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002)

    config = {"training": {"lr": 0.0002, "lambda_cycle": 10.0, "lambda_identity": 5.0}}

    pl_module = CycleGANLightningModule(
        G_AB=G_AB,
        G_BA=G_BA,
        D_A=D_A,
        D_B=D_B,
        optimizer_G=optimizer_G,
        optimizer_D_A=optimizer_D_A,
        optimizer_D_B=optimizer_D_B,
        config=config,
    )

    assert pl_module is not None
    assert pl_module.automatic_optimization is False

    batch = [
        torch.randn(1, 3, 64, 64),  # real_A
        torch.randn(1, 3, 64, 64),  # real_B
    ]

    # Run step
    pl_module.training_step(batch, 0)
    assert "loss" in pl_module.last_step_metrics
    assert "cycle_loss" in pl_module.last_step_metrics
