import os

from src.utils.remote_gpu import GPUManager


def download_models():
    mgr = GPUManager()
    mgr.add_backend_from_json("remote_gpu", "gpu_connection.json")
    with mgr.use("remote_gpu") as gpu:
        print("Downloading final checkpoints...")
        # Download into 'models/' so SCP places the remote 'checkpoints' folder
        # directly as 'models/checkpoints/', avoiding a nested double-directory.
        os.makedirs("models", exist_ok=True)
        gpu.download("/root/project/models/checkpoints", "models", recursive=True)


if __name__ == "__main__":
    download_models()
