import pytest
from unittest.mock import patch
from src.training.manager import TrainingManager


@pytest.fixture
def manager():
    return TrainingManager()


def test_manager_status_initial(manager):
    status = manager.get_status()
    assert status["is_running"] is False
    assert status["progress"] == 0


def test_manager_config_overrides(manager):
    # Mock _run_training to avoid any thread/process side effects
    with patch.object(TrainingManager, "_run_training"):
        # Explicitly set remote to false for local testing
        success, message = manager.start_training({"epochs": -1, "remote": False})
        assert success is True
        assert manager.is_running is True
        assert manager.current_run_id is not None
        
        # Cleanup
        manager.is_running = False


def test_manager_stop_when_not_running(manager):
    # Stop should return False if not running
    success, message = manager.stop_training()
    assert success is False
    assert "no training in progress" in message.lower()


def test_run_id_generation(manager):
    # Test if manager generates a run_id
    with patch.object(TrainingManager, "_run_training"):
        manager.start_training({})
        assert manager.current_run_id is not None
        assert manager.current_run_id.startswith("run_")
        manager.is_running = False
