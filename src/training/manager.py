"""
Orchestration layer for managing training processes and telemetry streaming.
Handles local/remote training lifecycle, log monitoring, and dashboard updates.
"""

import threading
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


from src.utils import get_training_config
from src.training.strategies import get_training_strategy

project_root = Path(__file__).parent.parent.parent


class TrainingManager:
    """
    Manages the training lifecycle and state for the API and Dashboard.
    Supports asynchronous process execution and real-time log parsing.
    """

    is_running: bool
    progress: float
    current_epoch: int
    total_epochs: int
    loss_history: List[Optional[float]]
    adv_loss_history: List[Optional[float]]
    log_history: List[str]
    status_message: str
    current_metrics: Dict[str, float]
    display_metadata: Dict[str, Any]
    current_run_id: Optional[str]
    last_run_id: Optional[str]
    _stop_event: threading.Event
    _thread: Optional[threading.Thread]
    frozen_config_path: Path

    def __init__(self) -> None:
        """Initializes the TrainingManager with default state."""
        self.is_running = False
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.loss_history = []
        self.adv_loss_history = []
        self.log_history = []
        self.status_message = "Idle"
        self.current_metrics = {}
        self.display_metadata = {}
        self.current_run_id = None
        self.last_run_id = self._detect_last_run_id()
        self._stop_event = threading.Event()
        self._thread = None

    def _detect_last_run_id(self) -> Optional[str]:
        """
        Scans results/runs for the most recently modified run folder.

        Returns:
            The run ID of the latest run, or None if no runs exist.
        """
        try:
            project_root_path = Path(__file__).parent.parent.parent
            runs_dir = project_root_path / "results" / "runs"
            if not runs_dir.exists():
                return None

            runs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not runs:
                return None

            # Sort by modification time of the directory
            latest_run = max(runs, key=lambda d: d.stat().st_mtime)
            return latest_run.name
        except Exception:
            return None

    def _get_initial_display_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Heuristically determine display metadata based on config before training starts.

        Args:
            config: Full project configuration.

        Returns:
            Metadata dictionary for the dashboard.
        """
        from src.utils.config_manager import get_display_metadata_for_config

        return get_display_metadata_for_config(config)

    def start_training(
        self, config_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Launches a training run in a background thread.

        Args:
            config_overrides: Configuration overrides from the user/API.

        Returns:
            A tuple of (success, message).
        """
        config_overrides = config_overrides or {}
        if self.is_running:
            return False, "Training already in progress"

        # 1. Start with full default config from disk
        from src.utils import load_config

        final_config: Dict[str, Any] = load_config()

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
            if config_overrides.get("config_path"):
                # Load special config WITHOUT user overrides to preserve YAML values
                special_cfg = load_config(
                    str(config_overrides["config_path"]), use_user_overrides=False
                )
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
            str(config_overrides.get("run_id"))
            if config_overrides.get("run_id")
            else (
                str(final_config.get("run_id"))
                if final_config.get("run_id")
                else f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        )
        self.last_run_id = self.current_run_id

        # 4. Freeze configuration for reproducibility
        run_dir = project_root / "results" / "runs" / self.current_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        frozen_config_path = run_dir / "frozen_config.json"

        with open(frozen_config_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4)

        self.frozen_config_path = frozen_config_path
        self._stop_event.clear()
        self.log_history = []
        self.progress = 0.0
        self.current_metrics = {}
        self.display_metadata = self._get_initial_display_metadata(final_config)
        self.total_epochs = int(final_config.get("training", {}).get("epochs", 0))

        # Load existing history if resuming
        is_resume = bool(
            final_config.get("training", {}).get("resume") or final_config.get("resume")
        )
        if is_resume:
            file_history_dict = self._load_history_dict()
            if file_history_dict:
                max_ep = max(file_history_dict.keys())
                self.loss_history = [None] * max_ep
                self.adv_loss_history = [None] * max_ep
                for ep, metrics in file_history_dict.items():
                    idx = ep - 1
                    self.loss_history[idx] = (
                        metrics.get("loss")
                        or metrics.get("train_loss")
                        or metrics.get("loss_pose")
                    )
                    self.adv_loss_history[idx] = metrics.get("adv_loss")
                self.current_epoch = max_ep
        else:
            self.loss_history = []
            self.adv_loss_history = []
            self.current_epoch = 0
            # Wipe local logs to prevent dashboard from loading old data
            for file_name in [
                "history.json",
                "stream.jsonl",
                "training.log",
                "evaluation.json",
            ]:
                file_path = run_dir / file_name
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Warning: could not delete {file_path}: {e}")

        self._thread = threading.Thread(target=self._run_training, args=(final_config,))
        self._thread.start()
        return True, "Training started"

    def stop_training(self) -> Tuple[bool, str]:
        """
        Sends a stop signal to the active training process.

        Returns:
            A tuple of (success, message).
        """
        if not self.is_running:
            return False, "No training in progress"

        self._stop_event.set()
        self.status_message = "Stopping..."
        return True, "Stop signal sent"

    def _handle_metrics_line(self, line: str) -> None:
        """Parse a dedicated metrics JSON line and update internal state."""
        try:
            # Expected format: [METRICS] {"epoch": 1, "loss": 0.5, ...}
            if "[METRICS]" not in line:
                return
            json_start = line.find("{")
            if json_start == -1:
                return
            json_str = line[json_start:]

            # Robust parsing: Sometimes lines are incomplete due to PTY line wrapping.
            # If so, json.loads will raise JSONDecodeError and the catch block will ignore it,
            # but the secondary clean stream (stream.jsonl) will provide the valid line a split-second later.
            metrics = json.loads(json_str)

            # Update current metrics
            for k, v in metrics.items():
                if k == "display_metadata":
                    self.display_metadata = v
                    self.log_history.append(
                        f"[{time.strftime('%H:%M:%S')}] [Manager] Received display metadata: {list(v.get('loss_labels', {}).keys())}"
                    )
                elif k not in ["epoch", "progress", "is_summary"]:
                    self.current_metrics[k] = v

            self.current_epoch = int(metrics.get("epoch", self.current_epoch))
            self.progress = float(metrics.get("progress", self.progress))

            if metrics.get("is_summary"):
                idx = self.current_epoch - 1
                if idx >= 0:
                    while len(self.loss_history) <= idx:
                        self.loss_history.append(None)
                        self.adv_loss_history.append(None)

                    loss_val = metrics.get("loss", metrics.get("loss_pose"))
                    if loss_val is not None:
                        self.loss_history[idx] = float(loss_val)
                    if "adv_loss" in metrics:
                        self.adv_loss_history[idx] = float(metrics["adv_loss"])

            # Update status message from metrics if available
            if "status" in metrics:
                self.status_message = str(metrics["status"])
            elif metrics.get("is_summary"):
                self.status_message = f"Epoch {self.current_epoch} complete"
        except Exception as e:
            print(f"[TrainingManager] Error parsing metrics stream: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Retrieves the current training status and metrics.
        Synchronizes with disk history to ensure accuracy.

        Returns:
            Dictionary containing running state, progress, and history.
        """
        # Always try to restore history from disk if in-memory lists are empty
        # but we have a valid run to look at.
        run_id = self.current_run_id or self.last_run_id

        if not self.loss_history and run_id:
            file_history_dict = self._load_history_dict()
            if file_history_dict:
                max_ep = max(file_history_dict.keys())
                # Initialize lists to correct size
                self.loss_history = [None] * max_ep
                self.adv_loss_history = [None] * max_ep
                for ep, metrics in file_history_dict.items():
                    idx = ep - 1
                    if 0 <= idx < len(self.loss_history):
                        self.loss_history[idx] = (
                            metrics.get("loss")
                            or metrics.get("train_loss")
                            or metrics.get("loss_pose")
                        )
                        self.adv_loss_history[idx] = metrics.get("adv_loss")

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
                try:
                    idx = ep - 1
                    if 0 <= idx < len(self.loss_history):
                        self.loss_history[idx] = (
                            metrics.get("loss")
                            or metrics.get("train_loss")
                            or metrics.get("loss_pose")
                        )
                        self.adv_loss_history[idx] = metrics.get("adv_loss")
                except (ValueError, TypeError):
                    continue

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
            "progress": 1.0
            if (not self.is_running and self.total_epochs > 0)
            else overall_progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "loss_history": self.loss_history,
            "adv_loss_history": self.adv_loss_history,
            "history_dict": file_history_dict,
            "log_history": self.log_history,
            "status_message": self.status_message,
            "current_metrics": self.current_metrics,
            "display_metadata": self.display_metadata,
        }

    def _load_history_dict(self) -> Dict[int, Dict[str, Optional[float]]]:
        """Loads history from disk and returns a dict mapping explicit epoch number to metrics."""
        try:
            project_root_path = Path(__file__).parent.parent.parent
            run_id = self.current_run_id or self.last_run_id
            if run_id:
                history_path = (
                    project_root_path / "results" / "runs" / run_id / "history.json"
                )
                if history_path.exists():
                    with open(history_path, "r", encoding="utf-8") as f:
                        history: List[Dict[str, Any]] = json.load(f)
                        result: Dict[int, Dict[str, Optional[float]]] = {}
                        import math

                        for i, entry in enumerate(history):
                            # Fallback to index-based epoch if 'epoch' key is missing
                            ep: int = int(entry.get("epoch", i + 1))

                            def sanitize(val: Any) -> Optional[float]:
                                try:
                                    if val is None:
                                        return None
                                    f_val = float(val)
                                    return (
                                        None
                                        if math.isnan(f_val) or math.isinf(f_val)
                                        else f_val
                                    )
                                except Exception:
                                    return None

                            # Store all metrics for this epoch
                            result[ep] = {
                                k: sanitize(v) for k, v in entry.items() if k != "epoch"
                            }
                        return result
        except Exception as e:
            print(f"[TrainingManager] Error loading history: {e}")
        return {}

    def _load_history_dual(self) -> Tuple[List[float], List[float]]:
        try:
            project_root_path = Path(__file__).parent.parent.parent
            if self.current_run_id:
                history_path = (
                    project_root_path
                    / "results"
                    / "runs"
                    / self.current_run_id
                    / "history.json"
                )
            else:
                history_path = (
                    project_root_path / "models" / "checkpoints" / "history.json"
                )

            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Support multiple loss keys for robustness
                    train_losses = []
                    for entry in data:
                        loss_val = (
                            entry.get("loss")
                            or entry.get("loss_pose")
                            or entry.get("train_loss")
                            or 0
                        )
                        train_losses.append(float(loss_val))

                    adv_losses = [float(entry.get("adv_loss", 0)) for entry in data]
                    return train_losses, adv_losses
        except Exception as e:
            print(f"[TrainingManager] Error loading history: {e}")
        return [], []

    def _run_training(self, config_overrides: Dict[str, Any]) -> None:
        try:
            self.status_message = "Initializing..."
            project_root_path = Path(__file__).parent.parent.parent

            # Extract remote flag from multiple possible locations in the config
            is_remote = False
            if config_overrides:
                remote_cfg = config_overrides.get("remote", {})
                if isinstance(remote_cfg, dict):
                    is_remote = bool(remote_cfg.get("use_remote", False))
                else:
                    is_remote = bool(remote_cfg)

            max_retries = 5
            retry_count = 0

            while retry_count <= max_retries:
                if self._stop_event.is_set():
                    break

                strategy = get_training_strategy(config_overrides)
                is_resume = (
                    retry_count > 0
                    or bool(config_overrides.get("training", {}).get("resume", False))
                    or bool(config_overrides.get("resume", False))
                )

                if is_remote:
                    if retry_count > 0:
                        self.status_message = f"Reconnecting and resuming (attempt {retry_count}/{max_retries})..."
                    else:
                        self.status_message = "Starting remote training..."

                    cmd = [
                        sys.executable,
                        "-u",
                        str(project_root_path / "scripts" / "remote_train.py"),
                    ]

                    # Pass the script path relative to project root to remote_train.py
                    script_path = strategy.get_script_path(project_root_path)
                    cmd.extend(
                        [
                            "--script",
                            str(script_path.relative_to(project_root_path).as_posix()),
                        ]
                    )
                else:
                    self.status_message = "Starting local training..."
                    cmd = [
                        sys.executable,
                        "-u",
                        str(strategy.get_script_path(project_root_path)),
                    ]

                # Add common arguments via strategy
                cmd.extend(
                    strategy.get_args(config_overrides, self.current_run_id, is_resume)
                )

                # Use the frozen config for the run
                relative_config_path = self.frozen_config_path.relative_to(
                    project_root_path
                ).as_posix()
                cmd.extend(["--config", relative_config_path])

                print(
                    f"[TrainingManager] Using config: {relative_config_path} (Attempt {retry_count + 1})"
                )
                self.log_history.append(
                    f"[{time.strftime('%H:%M:%S')}] [Manager] Executing: {' '.join(cmd)}"
                )

                start_log_idx = len(self.log_history)
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(project_root_path),
                )
                self.log_history.append(
                    f"[{time.strftime('%H:%M:%S')}] [Manager] Process started (PID: {process.pid}, Attempt: {retry_count + 1})"
                )

                if process.stdout:
                    for line in process.stdout:
                        if self._stop_event.is_set():
                            process.terminate()
                            self.status_message = "Stopping training..."
                            break

                        line_str = line.strip()
                        if line_str:
                            # Robust JSON Metrics Stream (skip log history)
                            if "[METRICS]" in line_str:
                                self._handle_metrics_line(line_str)
                                continue

                            # Skip interactive progress bar noise to prevent disk writes/spam
                            if (
                                "%|" in line_str
                                or "it/s" in line_str
                                or "s/it" in line_str
                                or "\r" in line_str
                                or "█" in line_str
                            ):
                                continue

                            # Add to log history with timestamp
                            timestamp = time.strftime("%H:%M:%S")
                            log_line = f"[{timestamp}] {line_str}"
                            print(
                                f"[TrainingManager] {line_str}"
                            )  # For backend debugging
                            self.log_history.append(log_line)
                            if len(self.log_history) > 1000:
                                self.log_history.pop(0)

                            # Persistence: Write to run-specific log file
                            if self.current_run_id:
                                log_dir = (
                                    project_root_path
                                    / "results"
                                    / "runs"
                                    / self.current_run_id
                                )
                                log_dir.mkdir(parents=True, exist_ok=True)
                                with open(
                                    log_dir / "training.log", "a", encoding="utf-8"
                                ) as f:
                                    f.write(log_line + "\n")

                            # --- Meaningful Status Extraction (Legacy Fallback) ---
                            if (
                                "Epoch" in line_str
                                and "/" in line_str
                                and ":" not in line_str
                            ):
                                self.status_message = line_str.strip()
                            elif "Training complete" in line_str:
                                self.status_message = "Training complete"

                process.wait()

                if self._stop_event.is_set():
                    break

                if process.returncode == 0:
                    self.status_message = "Training complete. Starting evaluation..."
                    self.progress = 0.95  # Almost done

                    # Trigger evaluation
                    success = self._run_evaluation(is_remote, str(self.current_run_id))

                    if success:
                        self.status_message = "Finished"
                        self.progress = 1.0
                    else:
                        self.status_message = "Training finished, but evaluation failed"
                        self.progress = 1.0

                    # FINAL SYNC: Ensure in-memory history is perfect before is_running=False
                    file_history_dict = self._load_history_dict()
                    if file_history_dict:
                        max_ep = max(file_history_dict.keys())
                        while len(self.loss_history) < max_ep:
                            self.loss_history.append(None)
                            self.adv_loss_history.append(None)
                        for ep, metrics in file_history_dict.items():
                            idx = ep - 1
                            if 0 <= idx < len(self.loss_history):
                                self.loss_history[idx] = (
                                    metrics.get("loss")
                                    or metrics.get("train_loss")
                                    or metrics.get("loss_pose")
                                )
                                self.adv_loss_history[idx] = metrics.get("adv_loss")

                    # Successfully completed, so exit the retry loop
                    break
                else:
                    # Non-zero return code. Check for SSH/Paramiko connection drop keywords.
                    is_conn_error = False
                    error_msg = f"Failed (exit {process.returncode})"

                    connection_keywords = [
                        "paramiko",
                        "ssh",
                        "connection",
                        "banner",
                        "pipe",
                        "10053",
                        "10054",
                        "tunnel",
                        "eof",
                        "reset by peer",
                        "handshake",
                        "timeout",
                        "disconnected",
                        "closed by",
                        "dropped",
                    ]

                    found_keywords = []
                    attempt_logs = self.log_history[start_log_idx:]
                    for log_line in reversed(attempt_logs):
                        line_lower = log_line.lower()
                        # Capture the last traceback or error statement
                        if (
                            "error:" in line_lower
                            or "exception:" in line_lower
                            or "filenotfounderror:" in line_lower
                        ):
                            if "Failed (exit" in error_msg or error_msg.startswith(
                                "Failed (exit"
                            ):
                                clean_err = (
                                    log_line.split("] ", 1)[-1]
                                    if "] " in log_line
                                    else log_line
                                )
                                error_msg = f"Error: {clean_err}"

                        # Match connection issues
                        for kw in connection_keywords:
                            if kw in line_lower:
                                is_conn_error = True
                                found_keywords.append(kw)

                    # Only attempt recovery if running on remote GPU and connection failed
                    if is_remote and is_conn_error and retry_count < max_retries:
                        retry_count += 1
                        warn_msg = (
                            f"[TrainingManager] Connection error detected (keywords: {list(set(found_keywords))}). "
                            f"Retrying in 15 seconds... (Attempt {retry_count}/{max_retries})"
                        )
                        print(warn_msg)
                        self.log_history.append(
                            f"[{time.strftime('%H:%M:%S')}] {warn_msg}"
                        )
                        self.status_message = f"Connection drop. Retrying {retry_count}/{max_retries} in 15s..."

                        # Wait for 15 seconds, checking stop event occasionally
                        for _ in range(15):
                            if self._stop_event.is_set():
                                break
                            time.sleep(1)
                    else:
                        # Unrecoverable error (code bug, CUDA OOM, config error) or maximum retries exceeded
                        self.status_message = error_msg
                        break

        except Exception as e:
            self.status_message = f"Error: {str(e)}"
        finally:
            self.is_running = False

    def _run_evaluation(self, is_remote: bool, run_id: str) -> bool:
        """Runs evaluation script for a specific run_id."""
        try:
            project_root_path = Path(__file__).parent.parent.parent
            if is_remote:
                cmd = [
                    sys.executable,
                    str(project_root_path / "scripts" / "remote_evaluate.py"),
                    "--run_id",
                    run_id,
                ]
            else:
                # For local evaluation, we might want to use torchrun if multiple GPUs are available,
                # but for simplicity and common local dev, we use normal python.
                # scripts/evaluate.py handles DDP if RANK env is set.
                cmd = [
                    sys.executable,
                    str(project_root_path / "scripts" / "evaluate.py"),
                    "--run_id",
                    run_id,
                    "--save_json",
                    str(
                        project_root_path
                        / "results"
                        / "runs"
                        / run_id
                        / "evaluation.json"
                    ),
                ]

            print(f"[TrainingManager] Running evaluation: {' '.join(cmd)}")

            # Run evaluation synchronously (within the training thread)
            eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root_path),
            )

            if eval_process.stdout:
                for line in eval_process.stdout:
                    if self._stop_event.is_set():
                        eval_process.terminate()
                        return False

                    line_str = line.strip()
                    if line_str:
                        timestamp = time.strftime("%H:%M:%S")
                        self.log_history.append(f"[{timestamp}] [Eval] {line_str}")
                        if "PCK@" in line_str or "Mean PCK" in line_str:
                            self.status_message = f"Evaluating: {line_str}"

            eval_process.wait()
            return eval_process.returncode == 0
        except Exception as e:
            print(f"[TrainingManager] Evaluation error: {e}")
            return False


training_manager = TrainingManager()
