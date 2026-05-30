"""
Remote GPU verification script.
Run via remote_train.py or directly via SSH to confirm CUDA is accessible.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    print("=" * 60)
    print("GPU ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. System-level GPU check
    print("\n[1/4] nvidia-smi check:")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  ✅ GPU detected: {result.stdout.strip()}")
    else:
        print(f"  ❌ nvidia-smi failed: {result.stderr.strip()}")

    # 2. Environment variables
    print("\n[2/4] Relevant environment variables:")
    for var in ["PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CONDA_PREFIX"]:
        val = os.environ.get(var, "<not set>")
        # Truncate PATH for readability
        if var == "PATH" and len(val) > 100:
            val = val[:100] + "..."
        print(f"  {var} = {val}")

    # 3. Python executable being used
    print(f"\n[3/4] Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")

    # 4. PyTorch CUDA check
    print("\n[4/4] PyTorch CUDA check:")
    try:
        import torch

        print(f"  torch version   : {torch.__version__}")
        print(f"  CUDA available  : {torch.cuda.is_available()}")
        print(f"  CUDA compiled   : {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"  Device name     : {torch.cuda.get_device_name(0)}")
            print(f"  Device count    : {torch.cuda.device_count()}")
            # Actually run a tensor op on the GPU to confirm it works
            x = torch.tensor([1.0]).cuda()
            print(f"  Tensor on GPU   : {x.device} ✅")
        else:
            # Dig into why
            try:
                torch.zeros(1).cuda()
            except Exception as e:
                print(f"  CUDA error      : {e}")
    except ImportError:
        print("  ❌ torch not installed")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
