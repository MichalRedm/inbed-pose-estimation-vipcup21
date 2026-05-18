"""
remote_train.py — Launch a training run on any remote GPU.

Works with any provider supported by gpu_connection.json:
  - Cloudflare tunnel (Kaggle, self-hosted): "type": "cloudflare_tunnel"
  - Direct SSH (RunPod, Vast.ai, Lambda Labs): "type": "ssh"

The remote environment is NOT manually configured here — all PATH /
LD_LIBRARY_PATH / CUDA setup lives in ~/.bash_profile on the remote server,
which is sourced automatically by the login shell (bash -l) used in every
gpu.run() call.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.remote_gpu import GPUManager
import argparse


def main():
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
        "--uda", action="store_true", help="Run Adversarial Domain Adaptation training"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation immediately after training"
    )
    args_cli, other_args = parser.parse_known_args()

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

    # Use the SSH key from the standard location
    ssh_key = os.path.expandvars(r"%USERPROFILE%\.ssh\id_ed25519")
    if os.path.exists(ssh_key):
        mgr._backends[backend_name].ssh_key = ssh_key

    print("--- Starting Remote Training Session ---")
    with mgr.use(backend_name) as gpu:
        # 1. Fresh Run Cleanup (Move BEFORE sync to avoid deleting uploaded configs)
        if not args_cli.resume and args_cli.run_id:
            remote_run_dir = f"/root/project/results/runs/{args_cli.run_id}"
            print(f"[clean] Wiping remote directory {remote_run_dir}...")
            gpu.run(f"rm -rf {remote_run_dir} || true", stream=False)
        elif not args_cli.resume:
            remote_ckpt_dir = "/root/project/models/checkpoints"
            print(f"[clean] Wiping remote checkpoints in {remote_ckpt_dir}...")
            gpu.run(f"rm -rf {remote_ckpt_dir}/* || true", stream=False)

        # 2. Sync local code and configs to remote
        gpu.sync_project(remote_dir="/root/project")

        # 3. Only pass project-specific env vars; PATH/CUDA come from login shell
        k_user = os.getenv("KAGGLE_USERNAME", "")
        k_key = os.getenv("KAGGLE_API_TOKEN", os.getenv("KAGGLE_KEY", ""))
        env_setup = (
            f"export KAGGLE_USERNAME={k_user} && "
            f"export KAGGLE_KEY={k_key} && "
            "export PYTHONPATH=$PYTHONPATH:."
        )

        # --- Step 1: GPU verification ---
        print("Verifying GPU on remote...")
        gpu.run(
            f"cd /root/project && {env_setup} && "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
            "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), "
            "'| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
        )

        # --- Step 2: Ensure data is present ---
        print("\nEnsuring data is available on remote...")
        download_cmd = "pip install kaggle -q && python3 scripts/download_dataset.py"
        gpu.run(f"cd /root/project && {env_setup} && {download_cmd}")

        # --- Step 3: Run training with incremental checkpoint sync ---
        print("\nChecking for multi-GPU setup...")
        gpu_count_res = gpu.run("nvidia-smi -L | wc -l", stream=False)
        try:
            detected_gpus = int(gpu_count_res.stdout.strip())
        except Exception:
            detected_gpus = 1

        num_gpus = detected_gpus
        if args_cli.max_gpus is not None:
            num_gpus = min(num_gpus, args_cli.max_gpus)

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

        training_script = "scripts/train.py"
        uda_flag = "--uda" if args_cli.uda else ""
        cmd = (
            f"cd /root/project && {env_setup} && "
            f"torchrun --nproc_per_node={num_gpus} --master_port={master_port} "
            f"{training_script} --data_root /root/project/data/raw {resume_flag} {run_id_flag} {uda_flag} {passthrough}"
        )

        # --- Step 4: Smart Cleanup & State Tracking ---
        # Determine local and remote paths based on run_id
        if args_cli.run_id:
            local_run_dir = Path("results/runs") / args_cli.run_id
            local_ckpt_dir = local_run_dir / "checkpoints"
            local_history_path = local_run_dir / "history.json"
            local_config_path = local_run_dir / "config.json"
            remote_ckpt_dir = (
                f"/root/project/results/runs/{args_cli.run_id}/checkpoints"
            )
            remote_history_path = (
                f"/root/project/results/runs/{args_cli.run_id}/history.json"
            )
            remote_config_path = (
                f"/root/project/results/runs/{args_cli.run_id}/config.json"
            )
        else:
            local_ckpt_dir = Path("models/checkpoints")
            local_history_path = local_ckpt_dir / "history.json"
            local_config_path = local_ckpt_dir / "config.json"
            remote_ckpt_dir = "/root/project/models/checkpoints"
            remote_history_path = "/root/project/models/checkpoints/history.json"
            remote_config_path = "/root/project/models/checkpoints/config.json"

        # Track which checkpoints have already been downloaded
        downloaded: set[str] = set()

        os.makedirs(local_ckpt_dir, exist_ok=True)

        # Initial snapshot: mark existing remote checkpoints as 'downloaded'
        # so we don't pull down old data at the start.
        print("Taking initial snapshot of remote checkpoints...")
        res = gpu.run(
            f"ls {remote_ckpt_dir}/*.pth 2>/dev/null || true",
            stream=False,
        )
        for f in res.stdout.splitlines():
            fname = f.strip().split("/")[-1]
            if fname.endswith(".pth"):
                downloaded.add(fname)
        print(f"Ignored {len(downloaded)} existing remote checkpoints.")

        def poll_and_download(session):
            """Download any checkpoint not yet synced locally."""
            # 1. Sync .pth checkpoints
            result = session.run(
                f"ls {remote_ckpt_dir}/*.pth 2>/dev/null || true",
                stream=False,
            )
            remote_files = [
                f.strip()
                for f in result.stdout.splitlines()
                if f.strip().endswith(".pth")
            ]
            for remote_path in remote_files:
                fname = remote_path.split("/")[-1]
                # Always download best_model.pth to ensure it's the latest
                if fname not in downloaded or fname == "best_model.pth":
                    print(f"\n[sync] Downloading {fname}...")
                    session.download(remote_path, str(local_ckpt_dir), recursive=False)
                    downloaded.add(fname)
                    print(f"[sync] {fname} saved to {local_ckpt_dir}/")

            # 2. Sync history.json and config.json
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

        # Run training in background thread; poll checkpoints and stream metrics from main thread
        import threading
        import time

        training_result: list = []

        def run_training():
            training_result.append(gpu.run(cmd))

        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        def run_polling():
            # Open a separate session for background polling
            try:
                with mgr.use(backend_name) as poll_session:
                    poll_and_download(poll_session)
                    poll_interval = 30
                    while training_thread.is_alive():
                        time.sleep(poll_interval)
                        try:
                            poll_and_download(poll_session)
                        except Exception as exc:
                            print(f"[sync] Warning: checkpoint poll failed: {exc}")
                    poll_and_download(poll_session)
            except Exception as e:
                print(f"[sync] Background poller crashed: {e}")

        def run_streaming():
            # Open a dedicated session for real-time metric streaming
            remote_stream_path = (
                f"/root/project/results/runs/{args_cli.run_id}/stream.jsonl"
            )
            print(f"[sync] Starting metrics streamer for {remote_stream_path}")

            while training_thread.is_alive():
                try:
                    with mgr.use(backend_name) as stream_session:
                        ssh = stream_session._ssh
                        # Run a python script on the remote to emulate tail -F but with explicit flushing.
                        # This avoids all pipe block-buffering issues inherent to `tail` over SSH without a PTY.
                        cmd = (
                            f"python -c '\n"
                            f"import time, os\n"
                            f'open("{remote_stream_path}", "a").close()\n'
                            f'f = open("{remote_stream_path}", "r")\n'
                            f"while True:\n"
                            f"    line = f.readline()\n"
                            f'    if line: print(line, end="", flush=True)\n'
                            f"    else: time.sleep(0.5)\n"
                            f"'"
                        )
                        _, stdout, _ = ssh.exec_command(cmd, get_pty=False)

                        # Set a timeout for reading to allow heartbeat/alive checks
                        stdout.channel.settimeout(10.0)

                        while training_thread.is_alive():
                            try:
                                line = stdout.readline()
                                if line:
                                    # Print with prefix for TrainingManager to intercept
                                    print(f"[METRICS] {line.strip()}", flush=True)
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
        poller_thread = threading.Thread(target=run_polling, daemon=True)
        streamer_thread = threading.Thread(target=run_streaming, daemon=True)

        poller_thread.start()
        streamer_thread.start()

        training_thread.join()
        poller_thread.join(timeout=60)

        result = training_result[0] if training_result else None
        if result is None or not result.ok():
            print("\nTraining failed. Stderr:")
            safe_stderr = (
                (result.stderr if result else "Unknown error")
                .encode(sys.stdout.encoding, errors="replace")
                .decode(sys.stdout.encoding)
            )
            print(safe_stderr)
            sys.exit(result.exit_code if result else 1)

        # --- Step 5: Automated Evaluation ---
        if args_cli.eval and args_cli.run_id:
            print("\n" + "=" * 40)
            print("--- Starting Automated Evaluation ---")
            print("=" * 40)

            eval_results_remote = (
                f"/root/project/results/runs/{args_cli.run_id}/evaluation.json"
            )
            eval_cmd = (
                f"cd /root/project && {env_setup} && "
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
                    remote_audit = f"/root/project/results/runs/{args_cli.run_id}/visual_audit_best_model.png"
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
