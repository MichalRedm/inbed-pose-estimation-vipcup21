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
        self.adv_loss_history: List[float] = []
        self.log_history: List[str] = []
        self.status_message = "Idle"
        self.current_metrics: Dict[str, float] = {}
        self.current_run_id: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_training(self, config_overrides: Optional[Dict] = None):
        if self.is_running:
            return False, "Training already in progress"

        # 1. Start with full default config from disk
        from src.utils import load_config

        final_config = load_config()

        # 2. Merge user_training.json (legacy/frontend settings)
        user_settings = get_training_config()
        if "training" not in final_config:
            final_config["training"] = {}

        # Only merge specific keys from user_settings to avoid overwriting everything
        for k in ["lr", "epochs", "batch_size", "augmentation"]:
            if k in user_settings:
                final_config["training"][k] = user_settings[k]

        if "remote" in user_settings:
            final_config["remote"] = {"use_remote": user_settings["remote"]}

        # 3. Apply passed overrides (highest priority)
        if config_overrides:
            # If the payload has a 'config_path', load it first
            if "config_path" in config_overrides:
                special_cfg = load_config(config_overrides["config_path"])
                final_config.update(special_cfg)

            # Then apply direct overrides
            if "training" in config_overrides:
                if "training" not in final_config:
                    final_config["training"] = {}
                final_config["training"].update(config_overrides["training"])

            # Handle other top-level keys (model, dataset, etc.)
            for k in ["model", "dataset", "remote", "uda", "run_id"]:
                if k in config_overrides:
                    if isinstance(config_overrides[k], dict) and k in final_config:
                        final_config[k].update(config_overrides[k])
                    else:
                        final_config[k] = config_overrides[k]

        self.is_running = True
        
        # Priority: 1. Payload run_id, 2. Config file run_id, 3. Timestamp
        self.current_run_id = (
            config_overrides.get("run_id") or 
            final_config.get("run_id") or 
            f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self._stop_event.clear()
        self.log_history = []
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0

        # Load existing history if resuming
        is_resume = final_config.get("training", {}).get("resume") or final_config.get(
            "resume"
        )
        if is_resume:
            self.loss_history, self.adv_loss_history = self._load_history_dual()
            if self.loss_history:
                self.current_epoch = len(self.loss_history)
        else:
            self.loss_history = []
            self.adv_loss_history = []

        self._thread = threading.Thread(target=self._run_training, args=(final_config,))
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
        file_history, file_adv_history = self._load_history_dual()
        if len(file_history) > len(self.loss_history):
            self.loss_history = file_history
            self.adv_loss_history = file_adv_history

        return {
            "is_running": self.is_running,
            "run_id": self.current_run_id,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "adv_loss_history": self.adv_loss_history,
            "log_history": self.log_history,
            "status_message": self.status_message,
            "current_metrics": self.current_metrics,
        }

    def _load_history_dual(self) -> tuple[List[float], List[float]]:
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
                    train_losses = [float(entry.get("train_loss", 0)) for entry in data]
                    adv_losses = [float(entry.get("adv_loss", 0)) for entry in data]
                    return train_losses, adv_losses
        except Exception as e:
            print(f"[TrainingManager] Error loading history: {e}")
        return [], []

    def _run_training(self, config_overrides):
        try:
            self.status_message = "Initializing..."
            project_root = Path(__file__).parent.parent.parent

            # Extract remote flag from multiple possible locations in the config
            is_remote = False
            if config_overrides:
                remote_cfg = config_overrides.get("remote", {})
                if isinstance(remote_cfg, dict):
                    is_remote = remote_cfg.get("use_remote", False)
                else:
                    is_remote = bool(remote_cfg)

            if is_remote:
                self.status_message = "Starting remote training..."
                cmd = [
                    sys.executable,
                    "-u",
                    str(project_root / "scripts" / "remote_train.py"),
                ]
            else:
                self.status_message = "Starting local training..."
                cmd = [sys.executable, "-u", str(project_root / "scripts" / "train.py")]

            if self.current_run_id:
                cmd.extend(["--run_id", self.current_run_id])

            # If a full config was provided, save it to a temporary file and pass it
            if config_overrides:
                # We'll save it in the configs/runs/RUN_ID dir (which is NOT excluded from sync)
                run_dir = project_root / "configs" / "runs" / self.current_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path = run_dir / "config.yaml"

                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(config_overrides, f)

                # Use relative POSIX path so it works on both local and remote (after sync)
                relative_config_path = config_path.relative_to(project_root).as_posix()
                cmd.extend(["--config", str(relative_config_path)])
                print(f"[TrainingManager] Using custom config: {relative_config_path}")

            self.log_history.append(
                f"[{time.strftime('%H:%M:%S')}] [Manager] Executing: {' '.join(cmd)}"
            )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
            )
            self.log_history.append(
                f"[{time.strftime('%H:%M:%S')}] [Manager] Process started (PID: {process.pid})"
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
                    log_line = f"[{timestamp}] {line}"
                    print(f"[TrainingManager] {line}")  # For backend debugging
                    self.log_history.append(log_line)
                    if len(self.log_history) > 1000:
                        self.log_history.pop(0)

                    # Persistence: Write to run-specific log file
                    if self.current_run_id:
                        log_dir = project_root / "results" / "runs" / self.current_run_id
                        log_dir.mkdir(parents=True, exist_ok=True)
                        with open(log_dir / "training.log", "a", encoding="utf-8") as f:
                            f.write(log_line + "\n")

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

                    # 4. Parse batch progress from tqdm: "Epoch 1/10:  20%|██        | 20/100 [00:10<00:40,  2.00it/s, loss=0.1234]"
                    tqdm_match = re.search(
                        r"Epoch \d+/\d+:\s+(\d+)%\|.*\| (\d+)/(\d+)(?:.*loss=([0-9.]+))?",
                        line,
                    )
                    if tqdm_match:
                        pct = tqdm_match.group(1)
                        curr_batch = tqdm_match.group(2)
                        total_batches = tqdm_match.group(3)
                        batch_loss = tqdm_match.group(4)

                        self.status_message = (
                            f"Progress: {pct}% (Batch {curr_batch}/{total_batches})"
                        )
                        
                        # Extract all metrics from tqdm postfix (e.g., key=value)
                        metrics = re.findall(r"(\w+)=([\d\.]+)", line)
                        for key, val in metrics:
                            try:
                                float_val = float(val)
                                self.current_metrics[key] = float_val

                                # Update primary loss history for live graphs
                                if key in ["loss", "train_loss"]:
                                    idx = self.current_epoch - 1
                                    if idx >= 0:
                                        while len(self.loss_history) <= idx:
                                            self.loss_history.append(0.0)
                                        self.loss_history[idx] = float_val
                                    if not self.status_message.endswith(f"Loss: {float_val:.4f}"):
                                        self.status_message += f" | Loss: {float_val:.4f}"
                                elif key == "adv_loss":
                                    idx = self.current_epoch - 1
                                    if idx >= 0:
                                        while len(self.adv_loss_history) <= idx:
                                            self.adv_loss_history.append(0.0)
                                        self.adv_loss_history[idx] = float_val
                            except ValueError:
                                continue
                        continue

                    # 5. Parse loss: "train_loss=0.1234" or "Epoch 1: loss=0.1234  adv_loss=0.05 val_loss=0.5678"
                    loss_match = re.search(
                        r"(?:Epoch (\d+): )?(?:train_)?loss=([0-9.]+)(?:\s+adv_loss=([0-9.]+))?(?:\s+(?:val_)?loss=([0-9.]+))?",
                        line,
                    )
                    if loss_match:
                        epoch_num = loss_match.group(1)
                        train_loss = float(loss_match.group(2))
                        adv_loss_str = loss_match.group(3)
                        val_loss = loss_match.group(4)

                        msg = f"Last loss: {train_loss:.4f}"
                        if adv_loss_str:
                            msg += f" | Adv: {float(adv_loss_str):.4f}"
                        if val_loss:
                            msg += f" | Val: {float(val_loss):.4f}"
                        self.status_message = msg

                        if epoch_num:
                            idx = int(epoch_num) - 1
                            # Ensure lists are long enough
                            for lst in [self.loss_history, self.adv_loss_history]:
                                while len(lst) <= idx:
                                    lst.append(0.0)
                            self.loss_history[idx] = train_loss
                            if adv_loss_str:
                                self.adv_loss_history[idx] = float(adv_loss_str)
                        else:
                            # Fallback to appending if no epoch number
                            if (
                                not self.loss_history
                                or self.loss_history[-1] != train_loss
                            ):
                                self.loss_history.append(train_loss)
                                if adv_loss_str:
                                    self.adv_loss_history.append(float(adv_loss_str))
                                else:
                                    self.adv_loss_history.append(0.0)
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
                    # Find the last meaningful error in log history
                    error_msg = f"Failed (exit {process.returncode})"
                    for line in reversed(self.log_history):
                        if (
                            "Error:" in line
                            or "Exception:" in line
                            or "FileNotFoundError:" in line
                        ):
                            clean_err = (
                                line.split("] ", 1)[-1] if "] " in line else line
                            )
                            error_msg = f"Error: {clean_err}"
                            break
                    self.status_message = error_msg

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
