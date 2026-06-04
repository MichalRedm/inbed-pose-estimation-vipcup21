import pytest
import math
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

    # Epoch 19: (mid-point cosine)
    progress = 19 / 39
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected_mid = 0.25 + (0.6 - 0.25) * cosine_decay
    assert module._get_current_confidence_threshold(19) == pytest.approx(
        expected_mid, rel=1e-5
    )

    # Epoch 39 (last epoch): Should be end threshold
    assert module._get_current_confidence_threshold(39) == pytest.approx(0.25, rel=1e-5)

    # Over epoch 39 (progress maxed at 1.0)
    assert module._get_current_confidence_threshold(45) == pytest.approx(0.25, rel=1e-5)


def test_ema_alpha_scheduling():
    config = {
        "training": {
            "epochs": 40,
            "ema_alpha_start": 0.99,
            "ema_alpha_end": 0.999,
        }
    }
    model = DummyModel()
    module = SelfTrainingLightningModule(model=model, config=config)

    # Epoch 0: Should be start alpha (0.99)
    assert module._get_current_ema_alpha(0) == pytest.approx(0.99, rel=1e-5)

    # Epoch 19: (mid-point cosine)
    progress = 19 / 39
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected_mid = 0.999 + (0.99 - 0.999) * cosine_val
    assert module._get_current_ema_alpha(19) == pytest.approx(expected_mid, rel=1e-5)

    # Epoch 39 (last epoch): Should be end alpha (0.999)
    assert module._get_current_ema_alpha(39) == pytest.approx(0.999, rel=1e-5)


def test_joint_specific_thresholds():
    config = {
        "training": {
            "epochs": 40,
        }
    }
    model = DummyModel()
    module = SelfTrainingLightningModule(model=model, config=config)

    base_thresh = 0.4
    joint_thresh = module._get_joint_specific_thresholds(base_thresh)

    # Verify shape
    assert joint_thresh.shape == (14,)

    # Verify expected values (Core: 1.0, Limbs: 0.85, Extremity: 0.70)
    # 0: R_Ankle (Extremity) -> 0.70 * 0.4 = 0.28
    assert joint_thresh[0].item() == pytest.approx(0.28, rel=1e-5)
    # 1: R_Knee (Limb) -> 0.85 * 0.4 = 0.34
    assert joint_thresh[1].item() == pytest.approx(0.34, rel=1e-5)
    # 2: R_Hip (Core) -> 1.0 * 0.4 = 0.40
    assert joint_thresh[2].item() == pytest.approx(0.40, rel=1e-5)


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
