import threading
import subprocess
import sys
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Optional


from src.utils import get_training_config


class TrainingManager:
    def __init__(self):
        self.is_running = False
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss_history: List[float] = []
        self.log_history: List[str] = []
        self.status_message = "Idle"
        self.current_run_id: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_training(self, config_overrides: Optional[Dict] = None):
        if self.is_running:
            return False, "Training already in progress"

        # Load saved config as baseline overrides if nothing is passed or to fill gaps
        base_overrides = get_training_config()

        # Merge passed overrides into base
        if config_overrides:
            # Handle possible nesting from frontend
            if "training" in config_overrides:
                base_overrides.update(config_overrides["training"])
                if "remote" in config_overrides:
                    base_overrides["remote"] = config_overrides["remote"]
                if "resume" in config_overrides:
                    base_overrides["resume"] = config_overrides["resume"]
            else:
                base_overrides.update(config_overrides)

        actual_overrides = base_overrides

        self.is_running = True
        self.current_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self._stop_event.clear()
        self.log_history = []
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0

        # Load existing history if resuming
        if actual_overrides and actual_overrides.get("resume"):
            self.loss_history = self._load_history()
            if self.loss_history:
                self.current_epoch = len(self.loss_history)
        else:
            self.loss_history = []

        self._thread = threading.Thread(
            target=self._run_training, args=(actual_overrides,)
        )
        self._thread.start()
        return True, "Training started"

    def stop_training(self):
        if not self.is_running:
            return False, "No training in progress"

        self._stop_event.set()
        self.status_message = "Stopping..."
        return True, "Stop signal sent"

    def get_status(self):
        # Refresh loss history from file if it's more complete than our in-memory version
        # (Especially useful for remote training where history.json is synced periodically)
        file_history = self._load_history()
        if len(file_history) > len(self.loss_history):
            self.loss_history = file_history

        return {
            "is_running": self.is_running,
            "run_id": self.current_run_id,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "log_history": self.log_history,
            "status_message": self.status_message,
        }

    def _load_history(self) -> List[float]:
        try:
            project_root = Path(__file__).parent.parent.parent
            if self.current_run_id:
                history_path = (
                    project_root
                    / "results"
                    / "runs"
                    / self.current_run_id
                    / "history.json"
                )
            else:
                history_path = project_root / "models" / "checkpoints" / "history.json"

            if history_path.exists():
                with open(history_path, "r") as f:
                    data = json.load(f)
                    return [float(entry.get("train_loss", 0)) for entry in data]
        except Exception as e:
            print(f"[TrainingManager] Error loading history: {e}")
        return []

    def _run_training(self, config_overrides):
        try:
            self.status_message = "Initializing..."
            project_root = Path(__file__).parent.parent.parent
            is_remote = (
                config_overrides.get("remote", False) if config_overrides else False
            )

            if is_remote:
                self.status_message = "Starting remote training..."
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "remote_train.py"),
                ]
            else:
                self.status_message = "Starting local training..."
                cmd = [sys.executable, str(project_root / "scripts" / "train.py")]

            # Pass overrides as CLI args (works for both local and remote scripts)
            if config_overrides:
                if "lr" in config_overrides:
                    cmd.extend(["--lr", str(config_overrides["lr"])])
                if "epochs" in config_overrides:
                    cmd.extend(["--epochs", str(config_overrides["epochs"])])
                if "batch_size" in config_overrides:
                    cmd.extend(["--batch_size", str(config_overrides["batch_size"])])
                if config_overrides.get("resume"):
                    cmd.append("--resume")

            if self.current_run_id:
                cmd.extend(["--run_id", self.current_run_id])

            print(f"  Executing training command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
            )

            for line in process.stdout:
                if self._stop_event.is_set():
                    process.terminate()
                    self.status_message = "Stopping training..."
                    break

                line = line.strip()
                if line:
                    # Add to log history with timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[TrainingManager] {line}")  # For backend debugging
                    self.log_history.append(f"[{timestamp}] {line}")
                    if len(self.log_history) > 1000:
                        self.log_history.pop(0)

                    # --- Meaningful Status Extraction ---

                    # 1. Initialization Stages
                    if "Using device:" in line:
                        device_name = (
                            line.split("Using device:")[1].split("(")[0].strip()
                        )
                        self.status_message = f"Initialized on {device_name}"
                        continue

                    if "Validation samples:" in line:
                        count = line.split("Validation samples:")[1].strip()
                        self.status_message = (
                            f"Dataset loaded: {count} validation samples"
                        )
                        continue

                    if "Loading checkpoint:" in line:
                        ckpt = (
                            line.split("Loading checkpoint:")[1].strip().split("/")[-1]
                        )
                        self.status_message = f"Resuming from {ckpt}"
                        continue

                    # 2. Parse initial message: "Starting training for 10 epochs (from epoch 31)..."
                    start_match = re.search(
                        r"Starting training for (\d+) epochs \(from epoch (\d+)\)", line
                    )
                    if start_match:
                        count = int(start_match.group(1))
                        start = int(start_match.group(2))
                        self.total_epochs = start + count - 1
                        self.current_epoch = start - 1
                        self.status_message = f"Session started: {count} epochs"
                        continue

                    # 3. Parse Epoch progress: "--- Epoch 31/40 ---"
                    epoch_match = re.search(r"(?:--- )?Epoch (\d+)/(\d+)", line)
                    if epoch_match:
                        current = int(epoch_match.group(1))
                        total = int(epoch_match.group(2))
                        self.current_epoch = current
                        self.total_epochs = total
                        self.progress = current / total if total > 0 else 0
                        self.status_message = f"Starting Epoch {current}..."  # "Epoch n / m" is in header, so just show start
                        continue

                    # 4. Parse batch progress from tqdm: "Epoch 1/10:  20%|██        | 20/100 [00:10<00:40,  2.00it/s]"
                    tqdm_match = re.search(
                        r"Epoch \d+/\d+:\s+(\d+)%\|.*\| (\d+)/(\d+)", line
                    )
                    if tqdm_match:
                        pct = tqdm_match.group(1)
                        curr_batch = tqdm_match.group(2)
                        total_batches = tqdm_match.group(3)
                        # More concise during training
                        self.status_message = (
                            f"Progress: {pct}% (Batch {curr_batch}/{total_batches})"
                        )
                        continue

                    # 5. Parse loss: "train_loss=0.1234" or "Epoch 1: train_loss=0.1234  val_loss=0.5678"
                    loss_match = re.search(
                        r"(?:Epoch (\d+): )?train_loss=([0-9.]+)(?:\s+val_loss=([0-9.]+))?",
                        line,
                    )
                    if loss_match:
                        epoch_num = loss_match.group(1)
                        train_loss = float(loss_match.group(2))
                        val_loss = loss_match.group(3)

                        msg = f"Last loss: {train_loss:.4f}"
                        if val_loss:
                            msg += f" | Val: {float(val_loss):.4f}"
                        self.status_message = msg

                        if epoch_num:
                            idx = int(epoch_num) - 1
                            # Ensure list is long enough
                            while len(self.loss_history) <= idx:
                                self.loss_history.append(0.0)
                            self.loss_history[idx] = train_loss
                        else:
                            # Fallback to appending if no epoch number
                            if (
                                not self.loss_history
                                or self.loss_history[-1] != train_loss
                            ):
                                self.loss_history.append(train_loss)
                        continue

                    # 6. Remote Sync & Prep
                    if "[sync] Downloading" in line:
                        fname = line.split("Downloading ")[1].split("...")[0]
                        self.status_message = f"Syncing {fname}..."
                        continue

                    if "Verifying GPU" in line:
                        self.status_message = "Verifying remote GPU..."
                        continue

                    if "Ensuring data" in line:
                        self.status_message = "Preparing remote dataset..."
                        continue

                    # Default fallback for other interesting lines
                    if not any(
                        x in line for x in ["|", "it/s", "s/it"]
                    ):  # Filter out noisy tqdm updates
                        self.status_message = (
                            line[:120] + "..." if len(line) > 123 else line
                        )

            process.wait()
            if not self._stop_event.is_set():
                if process.returncode == 0:
                    self.status_message = "Training complete. Starting evaluation..."
                    self.progress = 0.95  # Almost done

                    # Trigger evaluation
                    success = self._run_evaluation(is_remote, self.current_run_id)

                    if success:
                        self.status_message = "Finished"
                        self.progress = 1.0
                    else:
                        self.status_message = "Training finished, but evaluation failed"
                        self.progress = 1.0
                else:
                    self.status_message = f"Failed (exit {process.returncode})"

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
        finally:
            self.is_running = False

    def _run_evaluation(self, is_remote: bool, run_id: str) -> bool:
        """Runs evaluation script for a specific run_id."""
        try:
            project_root = Path(__file__).parent.parent.parent
            if is_remote:
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "remote_evaluate.py"),
                    "--run_id",
                    run_id,
                ]
            else:
                # For local evaluation, we might want to use torchrun if multiple GPUs are available,
                # but for simplicity and common local dev, we use normal python.
                # scripts/evaluate.py handles DDP if RANK env is set.
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "evaluate.py"),
                    "--run_id",
                    run_id,
                    "--save_json",
                    str(project_root / "results" / "runs" / run_id / "evaluation.json"),
                ]

            print(f"[TrainingManager] Running evaluation: {' '.join(cmd)}")

            # Run evaluation synchronously (within the training thread)
            eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
            )

            for line in eval_process.stdout:
                if self._stop_event.is_set():
                    eval_process.terminate()
                    return False

                line = line.strip()
                if line:
                    timestamp = time.strftime("%H:%M:%S")
                    self.log_history.append(f"[{timestamp}] [Eval] {line}")
                    if "PCK@" in line or "Mean PCK" in line:
                        self.status_message = f"Evaluating: {line}"

            eval_process.wait()
            return eval_process.returncode == 0
        except Exception as e:
            print(f"[TrainingManager] Evaluation error: {e}")
            return False


training_manager = TrainingManager()
