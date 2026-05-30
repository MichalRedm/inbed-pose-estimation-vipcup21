"""
remote_train.py — Launch a training run on any remote GPU.

Works with any provider supported by gpu_connection.json:
  - Cloudflare tunnel (Kaggle, self-hosted): "type": "cloudflare_tunnel"
  - Direct SSH (RunPod, Vast.ai, Lambda Labs): "type": "ssh"
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set, cast
from dotenv import load_dotenv

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.remote_gpu import GPUManager, GPUSession
import argparse


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_gpus", type=int, default=None, help="Maximum number of GPUs to use"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--run_id", type=str, default=None, help="Unique ID for this run"
    )
    parser.add_argument(
        "--script", type=str, default="scripts/train.py", help="Script to run on remote"
    )
    parser.add_argument(
        "--uda", action="store_true", help="Run Adversarial Domain Adaptation training"
    )
    parser.add_argument(
        "--cyclegan",
        action="store_true",
        help="Run CycleGAN domain translation training",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation immediately after training"
    )
    args_cli, other_args = parser.parse_known_args()

    # Extract config path from other_args list (as --config is not explicitly defined in argparse)
    config_path: Optional[str] = None
    if "--config" in other_args:
        idx = other_args.index("--config")
        if idx + 1 < len(other_args):
            config_path = other_args[idx + 1]
    else:
        for arg in other_args:
            if arg.startswith("--config="):
                config_path = arg.split("=", 1)[1]

    # Load config file to check for resume flag (enables seamless resume when started via API)
    config_resume = False
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".json"):
                    import json
                    cfg = json.load(f)
                else:
                    import yaml
                    cfg = yaml.safe_load(f)
                config_resume = bool(cfg.get("training", {}).get("resume", False) or cfg.get(
                    "resume", False
                ))
        except Exception as e:
            print(
                f"Warning: could not read config file {config_path} for resume check: {e}"
            )

    if config_resume:
        print("[remote_train] Config file indicates resume=True. Forcing resume mode.")
        args_cli.resume = True

    json_path = "gpu_connection.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        print("       For Kaggle: download it from the notebook output.")
        print("       For other providers: create it with type/host/port fields.")
        sys.exit(1)

    mgr = GPUManager()
    # Reads "type" from the JSON — works for both cloudflare_tunnel and direct SSH
    backend_name = "remote_gpu"
    mgr.add_backend_from_json(backend_name, json_path)

    print("--- Starting Remote Training Session ---")
    with mgr.use(backend_name) as gpu:
        # Determine remote project root dynamically based on who we logged in as
        home_res = gpu.run("echo $HOME", stream=False)
        remote_home = home_res.stdout.strip() if home_res.ok() else "/root"
        if not remote_home:
            remote_home = "/root"
        remote_project_dir = f"{remote_home}/project"
        print(f"Detected remote home directory: {remote_home}")
        print(f"Using remote project directory: {remote_project_dir}")
        # 1. Fresh Run Cleanup & Accumulation Prevention (Move BEFORE sync to avoid deleting uploaded configs)
        if args_cli.run_id:
            remote_run_dir = f"{remote_project_dir}/results/runs/{args_cli.run_id}"
            if not args_cli.resume:
                print(f"[clean] Wiping remote directory {remote_run_dir}...")
                gpu.run(f"rm -rf {remote_run_dir} || true", stream=False)
            else:
                # Clean up any stale .tmp files that could cause resume issues
                print(f"[clean] Removing stale .tmp files in {remote_run_dir}...")
                gpu.run(
                    f"find {remote_run_dir} -name '*.tmp' -delete || true", stream=False
                )

            # Clean up all other run directories to prevent accumulation of massive checkpoint files
            print(
                "[clean] Cleaning up all other run folders under results/runs/ to prevent disk space accumulation..."
            )
            gpu.run(
                f"find {remote_project_dir}/results/runs/ -maxdepth 1 -mindepth 1 -type d ! -name '{args_cli.run_id}' -exec rm -rf {{}} \\; || true",
                stream=False,
            )
        else:
            if not args_cli.resume:
                remote_ckpt_dir = f"{remote_project_dir}/models/checkpoints"
                print(f"[clean] Wiping remote checkpoints in {remote_ckpt_dir}...")
                gpu.run(f"rm -rf {remote_ckpt_dir}/* || true", stream=False)

        # 2. Sync local code and configs to remote
        gpu.sync_project(remote_dir=remote_project_dir)

        # 2.1 Manually upload config if it's in a directory that was ignored (e.g. results/)
        if config_path and os.path.exists(config_path):
            # Normalize path for remote (forward slashes)
            remote_cfg_path = f"{remote_project_dir}/{Path(config_path).as_posix()}"
            print(
                f"[sync] Manually uploading configuration: {config_path} -> {remote_cfg_path}"
            )
            gpu.run(f"mkdir -p {os.path.dirname(remote_cfg_path)}", stream=False)
            gpu.upload(config_path, remote_cfg_path, recursive=False)

        # 3. Only pass project-specific env vars; PATH/CUDA come from login shell
        k_user = str(os.getenv("KAGGLE_USERNAME", ""))
        k_key = str(os.getenv("KAGGLE_API_TOKEN", os.getenv("KAGGLE_KEY", "")))
        env_setup = (
            f"export KAGGLE_USERNAME={k_user} && "
            f"export KAGGLE_KEY={k_key} && "
            "export PYTHONUNBUFFERED=1 && "
            "export PYTHONPATH=$PYTHONPATH:."
        )

        # --- Step 1: GPU verification ---
        print("Verifying GPU on remote...")
        gpu.run(
            f"cd {remote_project_dir} && {env_setup} && "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
            "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), "
            "'| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
        )

        # --- Step 2: Ensure data is present ---
        print("\nEnsuring data is available on remote...")
        download_cmd = "pip install kaggle -q && python3 scripts/download_dataset.py"
        gpu.run(f"cd {remote_project_dir} && {env_setup} && {download_cmd}")

        # --- Step 3: Run training with incremental checkpoint sync ---
        print("\nChecking for multi-GPU setup...")
        gpu_count_res = gpu.run("nvidia-smi -L | wc -l", stream=False)
        try:
            detected_gpus = int(gpu_count_res.stdout.strip())
        except Exception:
            detected_gpus = 1

        num_gpus = detected_gpus
        if args_cli.max_gpus is not None:
            num_gpus = min(num_gpus, int(args_cli.max_gpus))

        print(f"Detected GPUs: {detected_gpus}. Using: {num_gpus}")

        print("\nExecuting training on remote GPU...")
        print("Checkpoints will be downloaded locally as they are saved.\n")

        # Use torchrun for both single and multi-GPU to keep consistency
        resume_flag = "--resume" if args_cli.resume else ""
        run_id_flag = f"--run_id {args_cli.run_id}" if args_cli.run_id else ""
        passthrough = " ".join(other_args)

        # Use a random master port to avoid EADDRINUSE (Address already in use)
        # when running multiple benchmarks or restarting quickly
        import random

        master_port = random.randint(20000, 29999)

        training_script: str = str(args_cli.script)

        # If running the standard train.py, use torchrun for potential DDP support.
        # Otherwise (e.g. CycleGAN), use plain python as those scripts might not support DDP yet.
        if "train.py" in training_script:
            uda_flag = "--uda" if args_cli.uda else ""
            cyclegan_flag = "--cyclegan" if args_cli.cyclegan else ""
            cmd = (
                f"cd {remote_project_dir} && {env_setup} && "
                f"torchrun --nproc_per_node={num_gpus} --master_port={master_port} "
                f"{training_script} --data_root {remote_project_dir}/data/raw {resume_flag} {run_id_flag} {uda_flag} {cyclegan_flag} {passthrough}"
            )
        else:
            cmd = (
                f"cd {remote_project_dir} && {env_setup} && "
                f"python3 {training_script} --data_dir {remote_project_dir}/data/raw {resume_flag} {run_id_flag} {passthrough}"
            )

        # --- Step 4: Smart Cleanup & State Tracking ---
        # Determine local and remote paths based on run_id
        if args_cli.run_id:
            local_run_dir = Path("results/runs") / str(args_cli.run_id)
            local_ckpt_dir = local_run_dir / "checkpoints"
            local_history_path = local_run_dir / "history.json"
            local_config_path = local_run_dir / "config.json"
            remote_ckpt_dir = (
                f"{remote_project_dir}/results/runs/{args_cli.run_id}/checkpoints"
            )
            remote_history_path = (
                f"{remote_project_dir}/results/runs/{args_cli.run_id}/history.json"
            )
            remote_config_path = (
                f"{remote_project_dir}/results/runs/{args_cli.run_id}/config.json"
            )
        else:
            local_ckpt_dir = Path("models/checkpoints")
            local_history_path = local_ckpt_dir / "history.json"
            local_config_path = local_ckpt_dir / "config.json"
            remote_ckpt_dir = f"{remote_project_dir}/models/checkpoints"
            remote_history_path = (
                f"{remote_project_dir}/models/checkpoints/history.json"
            )
            remote_config_path = f"{remote_project_dir}/models/checkpoints/config.json"

        # Track which checkpoints have already been downloaded
        downloaded: Set[str] = set()

        os.makedirs(local_ckpt_dir, exist_ok=True)

        # If resuming, check if local checkpoints exist and upload them to remote
        if args_cli.resume:
            print("\n[resume] Checking for local checkpoints to upload...")
            local_latest = local_ckpt_dir / "latest_model.pth"
            local_best = local_ckpt_dir / "best_model.pth"

            # Ensure remote checkpoint directory exists
            gpu.run(f"mkdir -p {remote_ckpt_dir}", stream=False)

            for local_path, fname in [
                (local_latest, "latest_model.pth"),
                (local_best, "best_model.pth"),
            ]:
                if local_path.exists():
                    # Verify integrity of local checkpoint to prevent uploading/overwriting with a corrupt file
                    is_valid = False
                    try:
                        print(f"[resume] Verifying integrity of local {fname}...")
                        if local_path.stat().st_size < 1000000:
                            print(
                                f"[resume] ERROR: local {fname} is too small ({local_path.stat().st_size} bytes)."
                            )
                        else:
                            import torch

                            with open(str(local_path), "rb") as f_ckpt:
                                torch.load(f_ckpt, map_location="cpu")
                            is_valid = True
                            print(
                                f"[resume] Local {fname} integrity verified successfully."
                            )
                    except Exception as integrity_err:
                        print(
                            f"[resume] ERROR: Local {fname} is corrupted: {integrity_err}"
                        )

                    if not is_valid:
                        print(
                            f"[resume] WARNING: Skipping upload of corrupted local {fname} to prevent remote corruption!"
                        )
                        if fname == "latest_model.pth":
                            raise RuntimeError(
                                f"Local {fname} is corrupted. Aborting resume upload to prevent overwriting healthy remote checkpoints."
                            )
                        continue

                    # Check if remote file exists and has the same size
                    remote_path = f"{remote_ckpt_dir}/{fname}"
                    try:
                        # Use sftp to check size
                        sftp_resume = gpu.open_sftp()
                        remote_stat = sftp_resume.stat(remote_path)
                        sftp_resume.close()
                        if remote_stat.st_size == local_path.stat().st_size:
                            print(
                                f"[resume] Remote {fname} is already up-to-date. Skipping upload."
                            )
                            continue
                    except Exception:
                        pass  # Remote file doesn't exist or error, proceed with upload

                    print(f"[resume] Uploading local {fname} to remote...")
                    try:
                        gpu.upload(
                            str(local_path),
                            f"{remote_ckpt_dir}/{fname}",
                            recursive=False,
                        )
                    except Exception as e:
                        print(
                            f"[resume] Warning: failed to upload {fname} due to {e}. Attempting to reconnect and retry..."
                        )
                        try:
                            gpu.disconnect()
                            gpu.connect()
                            gpu.upload(
                                str(local_path),
                                f"{remote_ckpt_dir}/{fname}",
                                recursive=False,
                            )
                            print(
                                f"[resume] Successfully uploaded {fname} after reconnect."
                            )
                        except Exception as retry_err:
                            print(
                                f"[resume] Warning: retry upload failed for {fname}: {retry_err}"
                            )
                            if fname == "latest_model.pth":
                                # latest_model.pth is strictly required for resume
                                raise retry_err
                else:
                    print(f"[resume] Local {fname} not found, skipping.")

        # Initial snapshot: mark existing remote checkpoints as 'downloaded'
        # so we don't pull down old data at the start.
        print("Taking initial snapshot of remote checkpoints...")
        res_ls = gpu.run(
            f"ls {remote_ckpt_dir}/*.pth 2>/dev/null || true",
            stream=False,
        )
        for f_ls in res_ls.stdout.splitlines():
            fname_ls = f_ls.strip().split("/")[-1]
            if fname_ls.endswith(".pth"):
                downloaded.add(fname_ls)
        print(f"Ignored {len(downloaded)} existing remote checkpoints.")

        def poll_metadata(session: GPUSession) -> None:
            """Sync history.json and config.json (Fast, updates dashboard)."""
            for r_path, l_path in [
                (remote_history_path, local_history_path),
                (remote_config_path, local_config_path),
            ]:
                # Use run_id in the temp filename to avoid collision/stale files in /tmp
                suffix = args_cli.run_id if args_cli.run_id else "default"
                tmp_remote = f"/tmp/sync_{suffix}_{os.path.basename(r_path)}"
                session.run(
                    f"if [ -f {r_path} ]; then cp {r_path} {tmp_remote}; else rm -f {tmp_remote}; fi",
                    stream=False,
                )
                try:
                    session.download(
                        tmp_remote,
                        str(l_path),
                        recursive=False,
                    )
                except Exception:
                    pass  # might not exist yet

        def poll_checkpoints(session: GPUSession) -> None:
            """Download any checkpoint not yet synced locally with strict size verification."""
            result_ls = session.run(
                f"ls {remote_ckpt_dir}/*.pth 2>/dev/null || true",
                stream=False,
            )
            remote_files = [
                f.strip()
                for f in result_ls.stdout.splitlines()
                if f.strip().endswith(".pth")
            ]

            sftp_poll: Optional[Any] = None
            try:
                sftp_poll = session.open_sftp()
            except Exception as sftp_err:
                print(
                    f"[sync] Warning: Could not open SFTP connection for verification: {sftp_err}"
                )

            for remote_path_poll in remote_files:
                fname_poll = remote_path_poll.split("/")[-1]
                # Always download best_model.pth and latest_model.pth to ensure they're complete
                is_key_model = fname_poll in ["best_model.pth", "latest_model.pth"]
                if fname_poll not in downloaded or is_key_model:
                    # Get remote size for integrity check
                    remote_size_poll: Optional[int] = None
                    if sftp_poll:
                        try:
                            remote_size_poll = sftp_poll.stat(remote_path_poll).st_size
                        except Exception:
                            pass

                    local_path_poll = local_ckpt_dir / fname_poll
                    temp_local_path_poll = local_ckpt_dir / f"{fname_poll}.tmp"

                    # Download with up to 3 retries and size verification
                    for attempt in range(1, 4):
                        print(
                            f"\n[sync] Downloading {fname_poll} (size={remote_size_poll} bytes, attempt {attempt})..."
                        )
                        try:
                            # Clean up old temp file if it exists to avoid partial write issues
                            if temp_local_path_poll.exists():
                                try:
                                    os.remove(temp_local_path_poll)
                                except OSError:
                                    pass

                            # Download to temporary file path
                            session.download(
                                remote_path_poll, str(temp_local_path_poll), recursive=False
                            )

                            # Verify local temp file exists and matches remote size
                            if temp_local_path_poll.exists():
                                local_size_poll = temp_local_path_poll.stat().st_size
                                if remote_size_poll is None or local_size_poll == remote_size_poll:
                                    # Verify checkpoint integrity for key model files
                                    is_valid_poll = True
                                    if fname_poll in ["best_model.pth", "latest_model.pth"]:
                                        try:
                                            import torch
                                            with open(str(temp_local_path_poll), "rb") as f_poll:
                                                torch.load(f_poll, map_location="cpu")
                                        except Exception as integrity_err_poll:
                                            print(
                                                f"[sync] Warning: Downloaded {fname_poll} failed integrity check: {integrity_err_poll}"
                                            )
                                            is_valid_poll = False

                                    if is_valid_poll:
                                        # Rename temp file to actual file atomically
                                        if local_path_poll.exists():
                                            try:
                                                os.remove(local_path_poll)
                                            except OSError:
                                                pass
                                        os.rename(temp_local_path_poll, local_path_poll)
                                        print(
                                            f"[sync] {fname_poll} successfully saved, verified, and renamed! ({local_size_poll} bytes)"
                                        )
                                        downloaded.add(fname_poll)
                                        break
                                    else:
                                        print(
                                            f"[sync] Warning: Checkpoint integrity check failed for {fname_poll}."
                                        )
                                else:
                                    print(
                                        f"[sync] Warning: Size mismatch for {fname_poll}! Remote: {remote_size_poll}, Local: {local_size_poll}"
                                    )
                            else:
                                print(
                                    f"[sync] Warning: Local temp file {fname_poll}.tmp not found after download."
                                )
                        except Exception as dl_err:
                            print(
                                f"[sync] Warning: Download failed for {fname_poll}: {dl_err}"
                            )
                        finally:
                            # Clean up temp file if rename didn't happen (i.e. on error/mismatch)
                            if temp_local_path_poll.exists():
                                try:
                                    os.remove(temp_local_path_poll)
                                except OSError:
                                    pass

                        time.sleep(2.0)
                    else:
                        print(
                            f"[sync] ERROR: Failed to download and verify {fname_poll} after 3 attempts!"
                        )

            if sftp_poll:
                try:
                    sftp_poll.close()
                except Exception:
                    pass

        # Run training in background thread; poll checkpoints and stream metrics from main thread
        import threading
        import time

        training_result: List[Any] = []

        def run_training() -> None:
            training_result.append(gpu.run(cmd))

        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        def run_metadata_polling() -> None:
            # Open a separate session for background metadata polling (extremely fast and lightweight)
            try:
                with mgr.use(backend_name) as metadata_session:
                    poll_metadata(metadata_session)
                    poll_interval = 5  # Poll metadata every 5 seconds for real-time dashboard updates!
                    while training_thread.is_alive():
                        time.sleep(poll_interval)
                        try:
                            poll_metadata(metadata_session)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[sync] Background metadata poller crashed: {e}")

        def run_checkpoint_polling() -> None:
            # Open a separate session for background checkpoint polling (heavier checks)
            try:
                with mgr.use(backend_name) as ckpt_session:
                    poll_checkpoints(ckpt_session)
                    poll_interval = 30  # Check checkpoints every 30 seconds
                    while training_thread.is_alive():
                        time.sleep(poll_interval)
                        try:
                            poll_checkpoints(ckpt_session)
                        except Exception as exc:
                            print(f"[sync] Warning: checkpoint poll failed: {exc}")
            except Exception as e:
                print(f"[sync] Background checkpoint poller crashed: {e}")

        def run_streaming() -> None:
            # Open a dedicated session for real-time metric streaming
            remote_stream_path_str = (
                f"{remote_project_dir}/results/runs/{args_cli.run_id}/stream.jsonl"
            )
            print(f"[sync] Starting metrics streamer for {remote_stream_path_str}")

            while training_thread.is_alive():
                try:
                    with mgr.use(backend_name) as stream_session:
                        # Run a python script on the remote to emulate tail -F but with explicit flushing.
                        # This avoids all pipe block-buffering issues inherent to `tail` over SSH without a PTY.
                        tail_cmd = (
                            f"python3 -c '\n"
                            f"import time, os\n"
                            f'open("{remote_stream_path_str}", "a").close()\n'
                            f'f = open("{remote_stream_path_str}", "r")\n'
                            f'buffer = ""\n'
                            f"while True:\n"
                            f"    chunk = f.read(1024)\n"
                            f"    if chunk:\n"
                            f"        buffer += chunk\n"
                            f'        while "\\n" in buffer:\n'
                            f'            line, buffer = buffer.split("\\n", 1)\n'
                            f'            print(line + "\\n", end="", flush=True)\n'
                            f"    else: time.sleep(0.5)\n"
                            f"'"
                        )
                        _, stdout_stream, _ = stream_session.exec_command(tail_cmd, get_pty=False)

                        # Set a timeout for reading to allow heartbeat/alive checks
                        stdout_stream.channel.settimeout(10.0)

                        while training_thread.is_alive():
                            try:
                                line_stream = stdout_stream.readline()
                                if line_stream:
                                    # Print with prefix for TrainingManager to intercept
                                    # Check if already prefixed to avoid double-tagging
                                    if line_stream.strip().startswith("[METRICS]"):
                                        print(line_stream.strip(), flush=True)
                                    else:
                                        print(f"[METRICS] {line_stream.strip()}", flush=True)
                                else:
                                    # Might be EOF if tail -F was interrupted
                                    break
                            except TimeoutError:
                                # Heartbeat to keep connection alive and show we are still polling
                                # TrainingManager ignores unknown [METRICS] JSON or non-JSON
                                # but the presence of output keeps the pipe fresh.
                                print(
                                    f"[sync] Streamer heartbeat (training_alive={training_thread.is_alive()})",
                                    flush=True,
                                )
                                continue
                except Exception as e:
                    if training_thread.is_alive():
                        print(
                            f"[sync] Streamer thread encountered error: {e}. Retrying in 5s..."
                        )
                        time.sleep(5)
                    else:
                        break

        # Start background helper threads
        metadata_poller_thread = threading.Thread(
            target=run_metadata_polling, daemon=True
        )
        ckpt_poller_thread = threading.Thread(
            target=run_checkpoint_polling, daemon=True
        )
        streamer_thread = threading.Thread(target=run_streaming, daemon=True)

        metadata_poller_thread.start()
        ckpt_poller_thread.start()
        streamer_thread.start()

        training_thread.join()
        metadata_poller_thread.join(timeout=10)
        ckpt_poller_thread.join(timeout=60)

        # FINAL STRIKE SYNCHRONOUS SYNC: Run one final, strict, synchronous verification sync on the main thread
        print("\n[sync] Running final strict verification sync...")
        try:
            print("[sync] Running final metadata sync (history.json, config.json)...")
            poll_metadata(gpu)
            print("[sync] Final metadata sync complete.")
        except Exception as sync_err:
            print(
                f"[sync] Warning: Final metadata sync encountered an error: {sync_err}"
            )

        try:
            print("[sync] Running final checkpoint sync...")
            poll_checkpoints(gpu)
            print("[sync] Final checkpoint sync complete.")
        except Exception as sync_err:
            print(
                f"[sync] Warning: Final checkpoint sync encountered an error: {sync_err}"
            )

        result_train = training_result[0] if training_result else None
        if result_train is None or not result_train.ok():
            print("\nTraining failed. Stderr:")
            safe_stderr = (
                (result_train.stderr if result_train else "Unknown error")
                .encode(sys.stdout.encoding, errors="replace")
                .decode(cast(str, sys.stdout.encoding))
            )
            print(safe_stderr)
            sys.exit(result_train.exit_code if result_train else 1)

        # --- Step 5: Automated Evaluation ---
        if args_cli.eval and args_cli.run_id:
            print("\n" + "=" * 40)
            print("--- Starting Automated Evaluation ---")
            print("=" * 40)

            eval_results_remote = (
                f"{remote_project_dir}/results/runs/{args_cli.run_id}/evaluation.json"
            )
            eval_cmd = (
                f"cd {remote_project_dir} && {env_setup} && "
                f"torchrun --nproc_per_node={num_gpus} --master_port={master_port + 1} "
                f"scripts/evaluate.py --run_id {args_cli.run_id} --save_json {eval_results_remote}"
            )

            eval_res = gpu.run(eval_cmd)
            if eval_res.ok():
                print("\n[sync] Downloading evaluation results...")
                local_eval_path = local_run_dir / "evaluation.json"
                try:
                    gpu.download(
                        eval_results_remote, str(local_eval_path), recursive=False
                    )
                    print(f"[sync] Evaluation results saved to {local_eval_path}")
                except Exception as e:
                    print(f"[sync] Error downloading evaluation results: {e}")

                # Also download visual audit image
                try:
                    remote_audit = f"{remote_project_dir}/results/runs/{args_cli.run_id}/visual_audit_best_model.png"
                    gpu.download(remote_audit, str(local_run_dir), recursive=False)
                    print(f"[sync] Visual audit image downloaded to {local_run_dir}/")
                except Exception:
                    pass
            else:
                print("\nEvaluation failed. Check stderr above.")

    print("--- Remote Training Session Complete ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] Remote training failed: {e}")
        if "10054" in str(e) or "banner" in str(e).lower():
            print("\nPRO TIP: This usually means your Cloudflare tunnel is broken.")
            print("1. Check if the Kaggle notebook is still running.")
            print(
                "2. Ensure you have the LATEST 'tunnel_hostname' in gpu_connection.json."
            )
            print("3. Use the dashboard Settings tab to 'Verify Connection'.")
        sys.exit(1)
