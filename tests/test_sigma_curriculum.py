import pytest
from src.training.standard_trainer import StandardTrainer
from unittest.mock import MagicMock

def test_sigma_curriculum_decay():
    """
    Verify that _get_current_sigma correctly schedules sigma decay.
    """
    config = {
        "training": {
            "epochs": 40,
            "sigma_start": 3.0,
            "sigma_end": 1.5
        }
    }
    
    # We mock out all dependencies of StandardTrainer except self.config, sigma_start, sigma_end
    # or we can construct it simply
    trainer = MagicMock(spec=StandardTrainer)
    trainer.config = config
    trainer.sigma_start = 3.0
    trainer.sigma_end = 1.5
    
    # Bind the method manually
    trainer._get_current_sigma = StandardTrainer._get_current_sigma.__get__(trainer, StandardTrainer)
    
    # 1. At epoch 0 (start), sigma should be exactly sigma_start
    assert trainer._get_current_sigma(0) == 3.0
    
    # 2. At 70% of training (epoch 28 of 40), sigma should reach exactly sigma_end
    assert trainer._get_current_sigma(28) == 1.5
    
    # 3. At 100% of training (epoch 40), sigma should still be sigma_end
    assert trainer._get_current_sigma(40) == 1.5
    
    # 4. At epoch 14 (halfway to 70%), sigma should be exactly midpoint (2.25)
    assert trainer._get_current_sigma(14) == 2.25


if __name__ == "__main__":
    print("Running sigma curriculum tests...")
    test_sigma_curriculum_decay()
    print("All sigma curriculum tests passed successfully!")
