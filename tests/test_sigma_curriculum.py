import pytest
import torch
import torch.nn as nn
from src.training.standard_trainer import StandardTrainer
from src.data.dataset import VIPCupDataset


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)


def test_sigma_curriculum_decay():
    # Setup mock config
    config = {
        "training": {
            "epochs": 40,
            "sigma_start": 3.0,
            "sigma_end": 1.5,
        }
    }

    # Initialize StandardTrainer with mock objects
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    device = torch.device("cpu")

    trainer = StandardTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
    )

    # Assert start, mid, end of curriculum
    # Epochs are 0-indexed internally in range(epochs)
    assert trainer._get_current_sigma(0) == pytest.approx(3.0)

    # At 70% of 40 epochs (epoch 28), it should reach sigma_end (1.5)
    assert trainer._get_current_sigma(28) == pytest.approx(1.5)
    assert trainer._get_current_sigma(35) == pytest.approx(1.5)

    # Intermediate step checking
    sigma_14 = trainer._get_current_sigma(14)  # half way through 70% (28 epochs)
    assert sigma_14 == pytest.approx(2.25)


def test_dataset_set_sigma():
    # Mock data root and initialize a simple dataset structure if mock,
    # but here we can just test if VIPCupDataset implements set_sigma and generate_heatmaps
    # dynamically responds to the updated sigma
    ds = VIPCupDataset(
        root=".",
        subjects=[],  # empty is fine, won't load anything if not needed
        modalities=["IR"],
        covers=["uncover"],
    )

    ds.set_sigma(3.0)
    assert ds.sigma == pytest.approx(3.0)

    ds.set_sigma(1.5)
    assert ds.sigma == pytest.approx(1.5)
