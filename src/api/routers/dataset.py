import io
import base64
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, Optional, Union

from src.data.augmentations import (
    DataAugmenter,
    get_available_augmentations,
    apply_custom_augmentations,
)
from src.utils import LSP_JOINT_NAMES
from src.training.manager import training_manager
from src.api.state import dataset_container, AugmentationApplyRequest

router = APIRouter()


@router.get("/dataset/stats")
async def get_dataset_stats() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "train": 0,
        "valid": 0,
        "modalities": ["IR", "RGB"],
        "covers": set(),
    }
    for dict_key, json_key in [("train", "train"), ("val", "valid")]:
        ds = dataset_container.get(dict_key)
        if not ds:
            continue
        summary[json_key] = len(ds)
        summary["total"] += len(ds)
        for sample in ds.samples:
            summary["covers"].add(sample["cover"])
    summary["covers"] = sorted(list(summary["covers"]))
    return summary


@router.get("/dataset/samples")
async def get_samples(
    split: str = "train",
    page: int = 1,
    limit: int = 20,
    cover: Optional[str] = None,
    subject: Optional[int] = None,
) -> Dict[str, Any]:
    ds = dataset_container.get(split)
    if not ds:
        raise HTTPException(status_code=404, detail="Split not found")
    filtered = []
    for i, sample in enumerate(ds.samples):
        if (cover and sample["cover"] != cover) or (
            subject and sample["subject"] != subject
        ):
            continue
        mod = (
            "IR"
            if "IR" in sample["image_paths"]
            else list(sample["image_paths"].keys())[0]
        )
        filtered.append(
            {
                "index": i,
                "id": f"{split}_{i}",
                "subject": sample["subject"],
                "cover": sample["cover"],
                "modalities": list(sample["image_paths"].keys()),
                "has_joints": any(j is not None for j in sample["joints"].values()),
                "image_path": str(sample["image_paths"][mod]),
                "modality": mod,
            }
        )
    start, end = (page - 1) * limit, page * limit
    return {
        "total": len(filtered),
        "page": page,
        "limit": limit,
        "samples": filtered[start:end],
    }


@router.get("/dataset/sample/{split}/{idx}")
async def get_sample_detail(split: str, idx: int) -> Dict[str, Any]:
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[idx]
    resolutions, joints_data = {}, {}
    for mod, path in sample["image_paths"].items():
        try:
            with Image.open(path) as img:
                resolutions[mod] = {"width": img.width, "height": img.height}
        except Exception:
            resolutions[mod] = {"width": 256, "height": 256}
        joints_data[mod] = (
            sample["joints"][mod][:2, :].T.tolist()
            if sample["joints"][mod] is not None
            else None
        )
    return {
        "id": idx,
        "split": split,
        "subject": sample["subject"],
        "cover": sample["cover"],
        "modalities": list(sample["image_paths"].keys()),
        "resolutions": resolutions,
        "joints_per_modality": joints_data,
    }


@router.get("/dataset/image/{split}/{idx}", response_model=None)
async def get_dataset_image(
    split: str, idx: int, modality: str = "IR", augment: bool = False
) -> Union[FileResponse, StreamingResponse]:
    ds = dataset_container.get(split)
    if not ds or idx >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[idx]
    image_path = sample["image_paths"].get(
        modality, list(sample["image_paths"].values())[0]
    )
    if not augment:
        return FileResponse(image_path)
    image = Image.open(image_path).convert("L" if modality == "IR" else "RGB")
    joints = sample["joints"].get(modality)
    train_config: Dict[str, Any] = (
        training_manager.config if hasattr(training_manager, "config") else {}
    )
    aug_cfg = train_config.get("training", {}).get("augmentation", {})
    augmenter = DataAugmenter(
        config={
            "enabled": True,
            "occlusion_prob": 1.0,
            "flip_prob": aug_cfg.get("flip_prob", 0.5),
            "rotation_range": aug_cfg.get("rotation_range", [-30, 30]),
            "scaling_range": aug_cfg.get("scaling_range", [0.8, 1.2]),
        }
    )
    aug_img, _ = augmenter(image, joints, is_ir=(modality == "IR"))
    buf = io.BytesIO()
    aug_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/augmentations")
async def list_augmentations() -> Dict[str, Any]:
    return {"augmentations": get_available_augmentations()}


@router.post("/augmentations/apply")
async def apply_augmentations_endpoint(
    request: AugmentationApplyRequest,
) -> Dict[str, Any]:
    ds = dataset_container.get(request.split)
    if not ds or request.index >= len(ds):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample = ds.samples[request.index]
    if request.modality not in sample["image_paths"]:
        raise HTTPException(status_code=404, detail="Modality not found")
    image = Image.open(sample["image_paths"][request.modality]).convert(
        "L" if request.modality == "IR" else "RGB"
    )
    aug_image, aug_joints = apply_custom_augmentations(
        image,
        sample["joints"].get(request.modality),
        request.augmentations,
        is_ir=(request.modality == "IR"),
        dataset_root=str(ds.root),
    )
    if torch.is_tensor(aug_image):
        aug_image = v2.functional.to_pil_image(aug_image)
    buf = io.BytesIO()
    aug_image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    joints_list = []
    if aug_joints is not None:
        for i in range(aug_joints.shape[1]):
            joints_list.append(
                {
                    "name": LSP_JOINT_NAMES[i] if i < len(LSP_JOINT_NAMES) else f"J{i}",
                    "x": float(aug_joints[0, i]),
                    "y": float(aug_joints[1, i]),
                    "vis": float(aug_joints[2, i]),
                }
            )
    return {
        "image": f"data:image/png;base64,{img_str}",
        "joints": joints_list,
        "original_size": {"width": image.width, "height": image.height},
    }
