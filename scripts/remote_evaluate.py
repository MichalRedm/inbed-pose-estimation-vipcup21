"""
remote_evaluate.py — Launch an evaluation run on any remote GPU.

Works with any provider supported by gpu_connection.json:
  - Cloudflare tunnel (Kaggle, self-hosted): "type": "cloudflare_tunnel"
  - Direct SSH (RunPod, Vast.ai, Lambda Labs): "type": "ssh"
"""

import os
import sys
from pathlib import Path
import argparse
import random
from dotenv import load_dotenv

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.remote_gpu import GPUManager


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to evaluate")
    parser.add_argument(
        "--max_gpus", type=int, default=None, help="Maximum number of GPUs to use"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename",
    )
    args_cli, other_args = parser.parse_known_args()

    json_path = "gpu_connection.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    mgr = GPUManager()
    backend_name = "remote_gpu"
    mgr.add_backend_from_json(backend_name, json_path)

    # Use the SSH key from the standard location
    ssh_key = os.path.expandvars(r"%USERPROFILE%\.ssh\id_ed25519")
    if os.path.exists(ssh_key):
        mgr._backends[backend_name].ssh_key = ssh_key

    print(f"--- Starting Remote Evaluation for {args_cli.run_id} ---")
    with mgr.use(backend_name) as gpu:
        # 1. Sync local code to remote
        gpu.sync_project(
            remote_dir="/root/project",
            exclude=[
                ".git",
                ".venv",
                "dashboard",
                "data",
                "__pycache__",
                ".pytest_cache",
                ".agents",
                ".ipynb_checkpoints",
                "results",
                "logs",
            ],
        )

        # 2. Env setup
        k_user = os.getenv("KAGGLE_USERNAME", "")
        k_key = os.getenv("KAGGLE_API_TOKEN", os.getenv("KAGGLE_KEY", ""))
        env_setup = (
            f"export KAGGLE_USERNAME={k_user} && "
            f"export KAGGLE_KEY={k_key} && "
            "export PYTHONPATH=$PYTHONPATH:."
        )

        # 3. GPU verification
        print("Verifying GPU on remote...")
        gpu.run(
            f"cd /root/project && {env_setup} && "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        )

        # 4. Ensure data is present
        print("\nEnsuring data is available on remote...")
        download_cmd = (
            "test -d data/raw/train || "
            "(pip install kaggle -q && python3 scripts/download_dataset.py)"
        )
        gpu.run(f"cd /root/project && {env_setup} && {download_cmd}")

        # 5. Multi-GPU detection
        gpu_count_res = gpu.run("nvidia-smi -L | wc -l", stream=False)
        try:
            detected_gpus = int(gpu_count_res.stdout.strip())
        except Exception:
            detected_gpus = 1

        num_gpus = detected_gpus
        if args_cli.max_gpus is not None:
            num_gpus = min(num_gpus, args_cli.max_gpus)

        print(f"Detected GPUs: {detected_gpus}. Using: {num_gpus}")

        # 6. Run evaluation
        master_port = random.randint(20000, 29999)
        remote_results_path = (
            f"/root/project/results/runs/{args_cli.run_id}/evaluation.json"
        )

        cmd = (
            f"cd /root/project && {env_setup} && export CHECKPOINT_NAME={args_cli.checkpoint_name} && "
            f"torchrun --nproc_per_node={num_gpus} --master_port={master_port} "
            f"scripts/evaluate.py --run_id {args_cli.run_id} --save_json {remote_results_path} "
            f"{' '.join(other_args)}"
        )

        print("\nExecuting evaluation on remote GPU...")
        result = gpu.run(cmd)

        if result.ok():
            # 7. Download results
            local_results_dir = Path("results/runs") / args_cli.run_id
            os.makedirs(local_results_dir, exist_ok=True)
            local_results_path = local_results_dir / "evaluation.json"

            print("\n[sync] Downloading evaluation results...")
            try:
                gpu.download(
                    remote_results_path, str(local_results_path), recursive=False
                )
                print(f"[sync] Results saved to {local_results_path}")
            except Exception as e:
                print(f"[sync] Error downloading results: {e}")

            # Also download the visual audit image if generated
            checkpoint_stem = Path(args_cli.checkpoint_name).stem
            remote_audit_path = f"/root/project/results/runs/{args_cli.run_id}/visual_audit_{checkpoint_stem}.png"
            local_audit_path = local_results_dir / f"visual_audit_{checkpoint_stem}.png"
            try:
                print(f"[sync] Downloading visual audit to {local_audit_path}...")
                gpu.download(
                    remote_audit_path, str(local_audit_path), recursive=False
                )
                print(f"[sync] Visual audit saved to {local_audit_path}")
            except Exception as e:
                print(f"[sync] Warning: Could not download visual audit: {e}")
        else:
            print("\nEvaluation failed.")
            if result.stdout:
                print("--- STDOUT ---")
                print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)
            sys.exit(1)

    print("--- Remote Evaluation Complete ---")


if __name__ == "__main__":
    main()
