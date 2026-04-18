import os
import sys
from src.utils.remote_gpu import GPUManager
from src.utils import load_config

def main():
    config = load_config()
    remote_cfg = config.get("remote", {})
    
    json_path = "gpu_connection.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please download it from Kaggle.")
        sys.exit(1)
        
    mgr = GPUManager()
    mgr.add_backend_from_json("kaggle", json_path)
    
    # Override SSH key path if it's in the default user location on Windows
    ssh_key = os.path.expandvars(r"%USERPROFILE%\.ssh\id_ed25519")
    if os.path.exists(ssh_key):
        mgr._backends["kaggle"].ssh_key = ssh_key

    print("--- Starting Remote Training Session ---")
    with mgr.use("kaggle") as gpu:
        # 1. Sync local code to remote
        gpu.sync_project(remote_dir="/root/project")
        
        # 2. Setup Environment and Data
        # We need Kaggle credentials on the remote to download the dataset
        from dotenv import load_dotenv
        load_dotenv()
        
        k_user = os.getenv("KAGGLE_USERNAME")
        k_key = os.getenv("KAGGLE_API_TOKEN") # Uses token as key
        
        # CONFIRMED paths from diagnostic:
        # - libcuda.so is at /usr/local/nvidia/lib64/ (NOT /usr/local/cuda/lib64/)
        # - nvidia-smi is at /opt/bin/nvidia-smi
        # - Python is at /usr/local/bin/python
        py = "/usr/local/bin/python"
        env_setup = (
            "export PATH=/usr/local/bin:/opt/bin:/usr/local/cuda/bin:$PATH && "
            "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && "
            "export CUDA_HOME=/usr/local/cuda && "
            f"export KAGGLE_USERNAME={k_user} && "
            f"export KAGGLE_KEY={k_key} && "
            "export PYTHONPATH=$PYTHONPATH:."
        )

        # --- Step 1: GPU verification ---
        print("Verifying GPU on remote...")
        gpu.run(
            f"cd /root/project && {env_setup} && "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
            f"{py} -c \"import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\""
        )

        # --- Step 2: Ensure data is present ---
        print("\nEnsuring data is available on remote...")
        download_cmd = (
            "test -d data/raw/train || "
            f"(pip install kaggle -q && {py} scripts/download_dataset.py)"
        )
        gpu.run(f"cd /root/project && {env_setup} && {download_cmd}")

        # --- Step 3: Run training ---
        print("\nExecuting training on remote GPU...")
        cmd = f"cd /root/project && {env_setup} && {py} -u scripts/train.py --data_root data/raw"
        result = gpu.run(cmd)

        if not result.ok():
            print("\nTraining failed. Stderr:")
            print(result.stderr)
            sys.exit(result.exit_code)

    print("--- Remote Training Session Complete ---")

if __name__ == "__main__":
    main()
