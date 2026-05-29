import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.manager import training_manager  # noqa: E402
from src.data.dataset import VIPCupDataset  # noqa: E402
from src.api.inference import inference_service  # noqa: E402
from src.api.state import dataset_container, runs_static_dir  # noqa: E402

# Import routers
from src.api.routers.config import router as config_router  # noqa: E402
from src.api.routers.training import router as training_router  # noqa: E402
from src.api.routers.dataset import router as dataset_router  # noqa: E402
from src.api.routers.predict import router as predict_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    train_config = (
        training_manager.config if hasattr(training_manager, "config") else {}
    )
    checkpoint_dir = Path(project_root) / "models" / "checkpoints"
    checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))
    if checkpoints:
        inference_service.load_model(str(checkpoints[-1]))
    try:
        dataset_cfg = train_config.get("dataset", {})
        root_path = project_root / dataset_cfg.get("root", "data/raw")
        dataset_container["train"] = VIPCupDataset(
            root=root_path,
            subjects=range(
                dataset_cfg.get("subjects_train", [1, 30])[0],
                dataset_cfg.get("subjects_train", [1, 30])[1] + 1,
            ),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="train",
        )
        dataset_container["val"] = VIPCupDataset(
            root=root_path,
            subjects=range(
                dataset_cfg.get("subjects_val", [81, 90])[0],
                dataset_cfg.get("subjects_val", [81, 90])[1] + 1,
            ),
            modalities=dataset_cfg.get("modalities", ["RGB", "IR"]),
            covers=["uncover", "cover1", "cover2"],
            split="valid",
        )
    except Exception as e:
        print(f"Error initializing datasets: {e}")

    yield

    # Shutdown logic (can be added here if needed)


app = FastAPI(
    title="In-Bed Pose Estimation API",
    description="API for predicting 14 human joints from in-bed images (RGB or IR).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from runs directory
runs_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/runs", StaticFiles(directory=str(runs_static_dir)), name="runs")


@app.get("/")
async def root():
    gpu_info = {"available": torch.cuda.is_available()}
    if gpu_info["available"]:
        gpu_info["name"] = torch.cuda.get_device_name(0)
        try:
            free, total = torch.cuda.mem_get_info(0)
            gpu_info["memory"] = {
                "free": free / (1024**3),
                "total": total / (1024**3),
                "used": (total - free) / (1024**3),
            }
        except Exception:
            pass

    return {
        "status": "online",
        "version": "1.0.0",
        "gpu": gpu_info,
    }


# Include sub-routers with OpenAPI tagging
app.include_router(config_router, tags=["Config"])
app.include_router(training_router, tags=["Training"])
app.include_router(dataset_router, tags=["Dataset"])
app.include_router(predict_router, tags=["Inference"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "src")],
    )
