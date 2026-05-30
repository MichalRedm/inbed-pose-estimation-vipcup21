import io
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Dict, Any, Optional, List, Union, Tuple, cast

from src.api.inference import inference_service
from src.utils import LSP_JOINT_NAMES
from src.api.state import project_root

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    checkpoint: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        checkpoint_path: Optional[Path] = None
        if run_id:
            checkpoint_path = (
                project_root
                / "results"
                / "runs"
                / run_id
                / "checkpoints"
                / (checkpoint or "best_model.pth")
            )
        elif model_name:
            checkpoint_path = project_root / "models" / "checkpoints" / Path(model_name)
        else:
            checkpoints = sorted(
                list((project_root / "models" / "checkpoints").glob("*.pth"))
            )
            if checkpoints:
                checkpoint_path = checkpoints[-1]
        if checkpoint_path and checkpoint_path.exists():
            inference_service.load_model(str(checkpoint_path))

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Determine channels from loaded model if possible
        in_channels = 1  # Default
        if inference_service._model:
            m = inference_service._model
            # Use standardized in_channels property if available
            if hasattr(m, "in_channels"):
                in_channels = int(getattr(m, "in_channels"))
            # Legacy fallback
            elif hasattr(m, "model") and hasattr(cast(Any, m).model, "in_channels"):
                in_channels = int(getattr(cast(Any, m).model, "in_channels"))

        image = image.convert("RGB" if in_channels == 3 else "L")
        orig_size = image.size
        model_size_list: List[int] = (
            inference_service._config.get("dataset", {}).get(
                "image_size", [256, 256]
            )
            if inference_service._config
            else [256, 256]
        )
        model_size = (model_size_list[0], model_size_list[1])
        img_resized = image.resize(model_size)
        img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
        if in_channels == 1:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        else:
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        predict_out = inference_service.predict(img_tensor)
        if isinstance(predict_out, tuple):
            preds = predict_out[0]
        else:
            preds = predict_out

        scale_x, scale_y = orig_size[0] / model_size[0], orig_size[1] / model_size[1]
        results = [
            {
                "joint": LSP_JOINT_NAMES[i] if i < len(LSP_JOINT_NAMES) else f"J{i}",
                "x": float(x) * scale_x,
                "y": float(y) * scale_y,
            }
            for i, (x, y) in enumerate(preds[0].cpu().numpy())
        ]
        return {
            "filename": file.filename,
            "original_size": {"width": orig_size[0], "height": orig_size[1]},
            "predictions": results,
        }
    except (UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"File must be an image: {str(e)}")
    except Exception as e:
        import traceback

        with open("api.log", "a") as log:
            log.write(f"  ERROR in predict: {str(e)}\n")
            log.write(traceback.format_exc() + "\n")
        raise HTTPException(status_code=500, detail=str(e))
