import threading
import time
from typing import Dict, List, Optional
import torch
from src.training.trainer import PoseTrainer
from src.utils import load_config
from src.models.hrnet import get_pose_net
from src.data.dataset import VIPCupDataset
from torch.utils.data import DataLoader

class TrainingManager:
    def __init__(self):
        self.is_running = False
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss_history: List[float] = []
        self.status_message = "Idle"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_training(self, config_overrides: Optional[Dict] = None):
        if self.is_running:
            return False, "Training already in progress"
        
        self.is_running = True
        self._stop_event.clear()
        self.loss_history = []
        self.progress = 0.0
        
        self._thread = threading.Thread(target=self._run_training, args=(config_overrides,))
        self._thread.start()
        return True, "Training started"

    def stop_training(self):
        if not self.is_running:
            return False, "No training in progress"
        
        self._stop_event.set()
        self.status_message = "Stopping..."
        return True, "Stop signal sent"

    def get_status(self):
        return {
            "is_running": self.is_running,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "status_message": self.status_message
        }

    def _run_training(self, config_overrides):
        try:
            self.status_message = "Initializing..."
            config = load_config()
            if config_overrides:
                # Basic merge
                for k, v in config_overrides.items():
                    if isinstance(v, dict) and k in config:
                        config[k].update(v)
                    else:
                        config[k] = v
            
            is_remote = config_overrides.get("remote", False) if config_overrides else False
            
            import subprocess
            import sys
            from pathlib import Path
            import re
            
            project_root = Path(__file__).parent.parent.parent
            
            if is_remote:
                self.status_message = "Starting remote training..."
                cmd = [sys.executable, str(project_root / "scripts" / "remote_train.py")]
            else:
                self.status_message = "Starting local training..."
                cmd = [sys.executable, str(project_root / "scripts" / "train.py")]
                
                # Pass overrides as CLI args
                if config_overrides:
                    if "lr" in config_overrides:
                        cmd.extend(["--lr", str(config_overrides["lr"])])
                    if "epochs" in config_overrides:
                        cmd.extend(["--epochs", str(config_overrides["epochs"])])
                    if "batch_size" in config_overrides:
                        cmd.extend(["--batch_size", str(config_overrides["batch_size"])])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root)
            )
            
            for line in process.stdout:
                if self._stop_event.is_set():
                    process.terminate()
                    self.status_message = "Stopping training..."
                    break
                
                line = line.strip()
                if line:
                    # Update status message with current log line
                    # But filter out tqdm junk if possible, or just show it
                    if len(line) > 100:
                        self.status_message = line[:97] + "..."
                    else:
                        self.status_message = line

                    # Parse progress: "Epoch 1/10"
                    epoch_match = re.search(r"Epoch (\d+)/(\d+)", line)
                    if epoch_match:
                        current = int(epoch_match.group(1))
                        total = int(epoch_match.group(2))
                        self.current_epoch = current
                        self.total_epochs = total
                        self.progress = current / total
                    
                    # Parse loss: "train_loss=0.1234"
                    loss_match = re.search(r"train_loss=([0-9.]+)", line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        self.loss_history.append(loss)
            
            process.wait()
            if not self._stop_event.is_set():
                if process.returncode == 0:
                    self.status_message = "Finished"
                    self.progress = 1.0
                else:
                    self.status_message = f"Failed (exit {process.returncode})"
                
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
        finally:
            self.is_running = False


training_manager = TrainingManager()
