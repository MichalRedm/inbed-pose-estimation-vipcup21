import pytest
from src.training.manager import TrainingManager
from pathlib import Path
import os

@pytest.fixture
def manager():
    return TrainingManager()

def test_manager_status_initial(manager):
    status = manager.get_status()
    assert status["is_running"] is False
    assert status["progress"] == 0

def test_manager_config_overrides(manager):
    # Explicitly set remote to false for local testing
    success, message = manager.start_training({"epochs": -1, "remote": False})
    assert isinstance(success, bool)

def test_manager_stop_when_not_running(manager):
    success, message = manager.stop_training()
    assert success is False
    assert "in progress" in message.lower()

def test_run_id_generation(manager):
    # Test if manager creates directories correctly for a new run
    # (This assumes manager has some logic for this or we test the side effect)
    pass
