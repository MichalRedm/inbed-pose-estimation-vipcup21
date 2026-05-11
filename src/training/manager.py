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
        self.last_run_id: Optional[str] = None
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
        self.last_run_id = self.current_run_id
        self._stop_event.clear()
        self.log_history = []
        self.progress = 0.0
        self.total_epochs = final_config.get("training", {}).get("epochs", 0)

        # Load existing history if resuming
        is_resume = final_config.get("training", {}).get("resume") or final_config.get(
            "resume"
        )
        if is_resume:
            file_history_dict = self._load_history_dict()
            if file_history_dict:
                max_ep = max(file_history_dict.keys())
                self.loss_history = [None] * max_ep
                self.adv_loss_history = [None] * max_ep
                for ep, metrics in file_history_dict.items():
                    idx = ep - 1
                    self.loss_history[idx] = metrics["loss"]
                    self.adv_loss_history[idx] = metrics["adv_loss"]
                self.current_epoch = max_ep
        else:
            self.loss_history = []
            self.adv_loss_history = []
            self.current_epoch = 0

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
        # Refresh loss history from file using explicit epoch indices to avoid desyncs
        # (Especially useful for remote training where history.json is synced periodically)
        file_history_dict = self._load_history_dict()
        
        # Ensure our in-memory list is long enough
        if file_history_dict:
            max_epoch_in_file = max(file_history_dict.keys())
            while len(self.loss_history) < max_epoch_in_file:
                self.loss_history.append(None)
                self.adv_loss_history.append(None)
                
            # Merge disk history into in-memory array at explicit indices
            for ep, metrics in file_history_dict.items():
                idx = ep - 1
                if idx >= 0:
                    self.loss_history[idx] = metrics["loss"]
                    self.adv_loss_history[idx] = metrics["adv_loss"]

        # Calculate overall progress: (completed epochs + current epoch progress) / total epochs
        overall_progress = 0.0
        if self.total_epochs > 0:
            completed = max(0, self.current_epoch - 1)
            # If we are in the middle of an epoch, add the batch progress
            # Note: self.progress is the 0.0-1.0 progress within the CURRENT epoch
            overall_progress = (completed + self.progress) / self.total_epochs
            overall_progress = min(1.0, overall_progress)

        return {
            "is_running": self.is_running,
            "run_id": self.current_run_id or self.last_run_id,
            "progress": 1.0 if not self.is_running else overall_progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "adv_loss_history": self.adv_loss_history,
            "history_dict": file_history_dict,
            "log_history": self.log_history,
            "status_message": self.status_message if self.is_running else "Finished",
            "current_metrics": self.current_metrics,
        }

    def _load_history_dict(self) -> Dict[int, Dict[str, float]]:
        """Loads history from disk and returns a dict mapping explicit epoch number to metrics."""
        try:
            project_root = Path(__file__).parent.parent.parent
            run_id = self.current_run_id or self.last_run_id
            if run_id:
                history_path = (
                    project_root
                    / "results"
                    / "runs"
                    / run_id
                    / "history.json"
                )
                if history_path.exists():
                    with open(history_path, "r") as f:
                        history = json.load(f)
                        result = {}
                        import math
                        for i, entry in enumerate(history):
                            # Fallback to index-based epoch if 'epoch' key is missing
                            ep = entry.get("epoch", i + 1)
                            
                            def sanitize(val):
                                try:
                                    f_val = float(val)
                                    return None if math.isnan(f_val) or math.isinf(f_val) else f_val
                                except:
                                    return None

                            result[ep] = {
                                "loss": sanitize(entry.get("loss", entry.get("train_loss", 0.0))),
                                "adv_loss": sanitize(entry.get("adv_loss", 0.0))
                            }
                        return result
        except Exception as e:
            print(f"[TrainingManager] Error loading history: {e}")
        return {}

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
                    # Support multiple loss keys for robustness
                    train_losses = []
                    for entry in data:
                        l = entry.get("loss") or entry.get("loss_pose") or entry.get("train_loss") or 0
                        train_losses.append(float(l))
                    
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
            
            # Propagate resume flag
            is_resume = config_overrides.get("training", {}).get("resume") or config_overrides.get("resume")
            if is_resume:
                cmd.append("--resume")

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

                    if "Resuming from global epoch" in line:
                        try:
                            ep_num = int(line.split("global epoch")[1].strip())
                            self.current_epoch = ep_num
                            self.status_message = f"Resuming from Epoch {ep_num}"
                        except:
                            pass
                        continue

                    if "Remote Training Session Complete" in line or "Training finished" in line:
                        self.status_message = "Training finished"
                        self.is_running = False
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

                    # 3. Parse batch progress from tqdm (Flexible matching)
                    is_tqdm = "%|" in line and "|" in line
                    if is_tqdm:
                        # Extract epoch if it's in the prefix (e.g., "Epoch 21/30: 10%|...")
                        epoch_prefix_match = re.search(r"Epoch (\d+)/(\d+)", line)
                        if epoch_prefix_match:
                            self.current_epoch = int(epoch_prefix_match.group(1))
                            self.total_epochs = int(epoch_prefix_match.group(2))

                        # Extract percentage and steps
                        prog_match = re.search(r"(\d+)%\|.*\| (\d+)/(\d+)", line)
                        if prog_match:
                            self.progress = float(prog_match.group(1)) / 100.0
                        
                        # Extract speed and ETA
                        speed_match = re.search(r",\s*([\d\.]+)it/s", line)
                        if speed_match: self.current_metrics["speed"] = speed_match.group(1)
                        
                        eta_match = re.search(r"<(\d+:\d+)", line)
                        if eta_match: self.current_metrics["eta"] = eta_match.group(1)

                    # 4. Extract ALL key=value pairs (supporting float, scientific, and percentage)
                    # This now runs for EVERY line, including epoch summaries
                    kv_matches = re.findall(r"([a-zA-Z_]\w*)=([\d\.]+(?:%|e-?\d+)?)", line)
                    if kv_matches:
                        import math
                        current_batch_metrics = {}
                        for k, v in kv_matches:
                            try:
                                # Strip % if present and convert to normalized float
                                v_clean = v.rstrip('%')
                                f_val = float(v_clean)
                                if v.endswith('%'):
                                    f_val /= 100.0

                                if math.isnan(f_val) or math.isinf(f_val):
                                    val = None
                                else:
                                    val = f_val
                            except:
                                val = v
                            
                            # Store in temporary dict first
                            current_batch_metrics[k] = val
                            
                            # If it's a progress bar (tqdm), store with prefix to avoid "double jump"
                            if is_tqdm:
                                self.current_metrics[f"batch_{k}"] = val
                            else:
                                # Finalized metric from summary line - update main dict
                                self.current_metrics[k] = val
                                
                        # 5. Update loss history ONLY if it's an epoch summary line
                        if "Epoch" in line and ":" in line and not is_tqdm:
                            epoch_summary_match = re.search(r"Epoch (\d+):", line)
                            if epoch_summary_match:
                                target_epoch = int(epoch_summary_match.group(1))
                                self.current_epoch = target_epoch # Ensure synchronization
                                
                                for k in ["loss", "loss_pose", "train_loss"]:
                                    if k in current_batch_metrics:
                                        val = current_batch_metrics[k]
                                        idx = target_epoch - 1
                                        if idx >= 0:
                                            # Pad history if there's a gap
                                            while len(self.loss_history) < idx:
                                                self.loss_history.append(None)
                                                self.adv_loss_history.append(None)
                                            
                                            if val is not None:
                                                if len(self.loss_history) <= idx:
                                                    self.loss_history.append(float(val))
                                                else:
                                                    self.loss_history[idx] = float(val)
                                        break
                        continue

                    # 4. Parse Epoch progress: "--- Epoch 31/40 ---"
                    epoch_match = re.search(r"(?:--- )?Epoch (\d+)/(\d+)", line)
                    if epoch_match:
                        current = int(epoch_match.group(1))
                        total = int(epoch_match.group(2))
                        self.current_epoch = current
                        self.total_epochs = total
                        # Only set progress to 0 if it's the exact header (no tqdm)
                        if ":" not in line:
                            self.progress = current / total if total > 0 else 0
                        self.status_message = f"Starting Epoch {current}..."  # "Epoch n / m" is in header, so just show start
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
