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

from dotenv import load_dotenv

from src.utils.remote_gpu import GPUManager


def main():
    load_dotenv()
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

        # --- Step 3: Run training (full cycle) ---
        print("\nExecuting training on remote GPU...")
        cmd = (
            f"cd /root/project && {env_setup} && "
            "python3 -u scripts/train.py --data_root data/raw"
        )
        result = gpu.run(cmd)

        if not result.ok():
            print("\nTraining failed. Stderr:")
            print(result.stderr)
            sys.exit(result.exit_code)

    print("--- Remote Training Session Complete ---")


if __name__ == "__main__":
    main()
