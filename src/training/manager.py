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
            
            if is_remote:
                self.status_message = "Starting remote training..."
                import subprocess
                import sys
                from pathlib import Path
                
                project_root = Path(__file__).parent.parent.parent
                
                # Command to run remote training
                cmd = [sys.executable, str(project_root / "scripts" / "remote_train.py")]
                
                # Add any training overrides as passthrough args if needed
                # For now, just basic run
                
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
                        self.status_message = "Stopping remote training..."
                        break
                    
                    line = line.strip()
                    if line:
                        self.status_message = line
                        # Try to parse epoch progress if visible in logs
                        if "Epoch" in line and "/" in line:
                            try:
                                # Very basic parsing: "Epoch 1/10"
                                parts = line.split("Epoch")[-1].split("/")[0].strip()
                                current = int(parts)
                                total = int(line.split("/")[-1].split()[0].strip())
                                self.current_epoch = current
                                self.total_epochs = total
                                self.progress = current / total
                            except:
                                pass
                
                process.wait()
                if not self._stop_event.is_set():
                    self.status_message = "Remote training finished" if process.returncode == 0 else f"Remote training failed (exit {process.returncode})"
                
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Simplified setup for demonstration
                self.status_message = "Loading model..."
                model_cfg = config.get("model", {}).get("hrnet", {})
                model = get_pose_net(model_cfg).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=config.get("training", {}).get("lr", 0.001))
                criterion = torch.nn.MSELoss()
                
                trainer = PoseTrainer(model, optimizer, criterion, device, config)
                self.total_epochs = trainer.epochs
                
                self.status_message = "Training..."
                for epoch in range(self.total_epochs):
                    if self._stop_event.is_set():
                        self.status_message = "Stopped by user"
                        break
                    
                    self.current_epoch = epoch + 1
                    # Simulate an epoch
                    time.sleep(1) 
                    fake_loss = 0.5 * (0.9 ** epoch) + (time.time() % 0.1)
                    self.loss_history.append(fake_loss)
                    self.progress = (epoch + 1) / self.total_epochs
                
                if not self._stop_event.is_set():
                    self.status_message = "Finished"
                
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
        finally:
            self.is_running = False


training_manager = TrainingManager()
