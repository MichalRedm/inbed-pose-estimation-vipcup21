import pytest
from src.training.self_training_lightning import SelfTrainingLightningModule
from torch import nn


class DummyModel(nn.Module):
    def forward(self, x, **kwargs):
        return x


def test_adaptive_confidence_curriculum():
    config = {
        "training": {
            "epochs": 40,
            "conf_threshold_start": 0.6,
            "conf_threshold_end": 0.25,
        }
    }
    model = DummyModel()
    module = SelfTrainingLightningModule(model=model, config=config)

    # Epoch 0: Should be start threshold
    assert module._get_current_confidence_threshold(0) == pytest.approx(0.6, rel=1e-5)

    # Epoch 19: (mid-point)
    expected_mid = 0.6 + (0.25 - 0.6) * (19 / 39)
    assert module._get_current_confidence_threshold(19) == pytest.approx(
        expected_mid, rel=1e-5
    )

    # Epoch 39 (last epoch): Should be end threshold
    assert module._get_current_confidence_threshold(39) == pytest.approx(0.25, rel=1e-5)

    # Over epoch 39 (progress maxed at 1.0)
    assert module._get_current_confidence_threshold(45) == pytest.approx(0.25, rel=1e-5)


def test_adaptive_confidence_backward_compatibility():
    config = {
        "training": {
            "epochs": 40,
            "confidence_threshold": 0.35,
        }
    }
    model = DummyModel()
    module = SelfTrainingLightningModule(model=model, config=config)

    # Should remain constant
    assert module._get_current_confidence_threshold(0) == pytest.approx(0.35, rel=1e-5)
    assert module._get_current_confidence_threshold(39) == pytest.approx(0.35, rel=1e-5)
