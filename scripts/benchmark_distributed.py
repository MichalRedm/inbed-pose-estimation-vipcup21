import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.remote_gpu import GPUManager


def main():
    load_dotenv()
    json_path = "gpu_connection.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    mgr = GPUManager()
    backend_name = "remote_gpu"
    mgr.add_backend_from_json(backend_name, json_path)

    ssh_key = os.path.expandvars(r"%USERPROFILE%\.ssh\id_ed25519")
    if os.path.exists(ssh_key):
        mgr._backends[backend_name].ssh_key = ssh_key

    print("--- Starting Distributed Training Benchmark ---")
    with mgr.use(backend_name) as gpu:
        # Sync project
        gpu.sync_project(remote_dir="/root/project")

        k_user = os.getenv("KAGGLE_USERNAME", "")
        k_key = os.getenv("KAGGLE_API_TOKEN", os.getenv("KAGGLE_KEY", ""))
        env_setup = f"export KAGGLE_USERNAME={k_user} && export KAGGLE_KEY={k_key} && export PYTHONPATH=$PYTHONPATH:."

        # Detect GPUs
        gpu_count_res = gpu.run("nvidia-smi -L | wc -l", stream=False)
        num_gpus = int(gpu_count_res.stdout.strip())
        print(f"Detected {num_gpus} GPUs on remote.")

        if num_gpus < 2:
            print("Error: Benchmark requires at least 2 GPUs to compare.")
            sys.exit(1)

        results = {}
        import random

        # Case 1: Single GPU
        print("\n>>> Phase 1: Single GPU Benchmark (2 epochs)...")
        start_t = time.time()
        port1 = random.randint(20000, 29999)
        res1 = gpu.run(
            f"cd /root/project && {env_setup} && "
            f"torchrun --nproc_per_node=1 --master_port={port1} scripts/train.py --data_root data/raw --epochs 2"
        )
        if not res1.ok():
            print(f"Error in Phase 1: {res1.stderr}")
            sys.exit(1)

        results["single_gpu"] = time.time() - start_t
        print(f"Single GPU Time: {results['single_gpu']:.2f}s")

        # Case 2: Multi-GPU (DDP)
        print(f"\n>>> Phase 2: Multi-GPU ({num_gpus} GPUs) Benchmark (2 epochs)...")
        start_t = time.time()
        port2 = random.randint(20000, 29999)
        res2 = gpu.run(
            f"cd /root/project && {env_setup} && "
            f"torchrun --nproc_per_node={num_gpus} --master_port={port2} scripts/train.py --data_root data/raw --epochs 2"
        )
        if not res2.ok():
            print(f"Error in Phase 2: {res2.stderr}")
            sys.exit(1)

        results["multi_gpu"] = time.time() - start_t
        print(f"Multi-GPU Time: {results['multi_gpu']:.2f}s")

        # Report
        print("\n" + "=" * 40)
        print("          BENCHMARK RESULTS")
        print("=" * 40)
        print(f"Single GPU: {results['single_gpu']:.2f}s")
        print(f"Multi-GPU ({num_gpus}):  {results['multi_gpu']:.2f}s")

        speedup = results["single_gpu"] / results["multi_gpu"]
        print(f"Speedup Factor: {speedup:.2f}x")
        print("=" * 40)


if __name__ == "__main__":
    main()
