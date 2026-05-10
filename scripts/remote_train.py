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
    parser.add_argument("--uda", action="store_true", help="Run Adversarial Domain Adaptation training")
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
        # 1. Sync local code to remote
        gpu.sync_project(remote_dir="/root/project")

        # 2. Only pass project-specific env vars; PATH/CUDA come from login shell
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
        download_cmd = (
            "test -d data/raw/train || "
            "(pip install kaggle -q && python3 scripts/download_dataset.py)"
        )
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

        training_script = "scripts/train_uda.py" if args_cli.uda else "scripts/train.py"
        cmd = (
            f"cd /root/project && {env_setup} && "
            f"torchrun --nproc_per_node={num_gpus} --master_port={master_port} "
            f"{training_script} --data_root /root/project/data/raw {resume_flag} {run_id_flag} {passthrough}"
        )

        # --- Step 4: Smart Cleanup & State Tracking ---
        if not args_cli.resume:
            print("[clean] Starting fresh run, wiping remote checkpoints...")
            gpu.run("rm -rf /root/project/models/checkpoints/* || true", stream=False)
        else:
            print("[resume] Resuming, preserving remote checkpoints...")

        # Track which checkpoints have already been downloaded
        downloaded: set[str] = set()

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

        # Run training in background thread; poll checkpoints from main thread
        import threading
        import time

        training_result: list = []

        def run_training():
            training_result.append(gpu.run(cmd))

        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        def run_polling():
            # Open a second session for background polling to avoid thread safety issues
            # with the primary training session.
            try:
                with mgr.use(backend_name) as poll_session:
                    # Sync initial history if it exists
                    poll_and_download(poll_session)

                    poll_interval = 30  # seconds between remote checkpoint checks
                    while training_thread.is_alive():
                        time.sleep(poll_interval)
                        try:
                            poll_and_download(poll_session)
                        except Exception as exc:
                            print(f"[sync] Warning: checkpoint poll failed: {exc}")

                    # Final sync to catch any checkpoint saved in the last polling window
                    poll_and_download(poll_session)
            except Exception as e:
                print(f"[sync] Background poller crashed: {e}")

        # Start background poller thread
        poller_thread = threading.Thread(target=run_polling, daemon=True)
        poller_thread.start()

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
