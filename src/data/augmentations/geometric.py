import random
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from PIL import Image
from typing import Optional, Tuple, List, Union, Dict, Any, cast


class HorizontalFlipAugmentation:
    """
    Randomly flips the image horizontally and reorders the joints for symmetry.
    """

    METADATA: Dict[str, Any] = {
        "id": "flip",
        "name": "Horizontal Flip",
        "order": 0,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "force_apply": {"type": "bool", "default": False},
        },
    }

    probability: float

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[Union[Image.Image, torch.Tensor], Optional[Any]]:
        prob = float(kwargs.get("probability", self.probability))
        force = bool(kwargs.get("force_apply", False))

        if not force and random.random() > prob:
            return image, joints

        # 1. Image Flip
        image = v2.functional.hflip(image)

        # 2. Joints Flip & Reorder
        if joints is not None:
            # Handle tv_tensors.Keypoints
            if torch.is_tensor(joints) and hasattr(joints, "canvas_size"):
                kpts = v2.functional.hflip(joints)
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                kpts = tv_tensors.Keypoints(
                    kpts[:, flip_indices, :], canvas_size=cast(tv_tensors.Keypoints, joints).canvas_size
                )
                return image, kpts

            # Manual reorder for numpy/raw tensor (3, 14)
            if hasattr(image, "width"):
                img_w = cast(Image.Image, image).width
            else:
                img_w = cast(torch.Tensor, image).shape[-1]

            import numpy as np
            if isinstance(joints, np.ndarray):
                joints_np = joints.copy()
                joints_np[0, :] = img_w - joints_np[0, :]  # Flip X
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                joints_np = joints_np[:, flip_indices]
                return image, joints_np

        return image, joints


class AffineAugmentation:
    """
    Applies random affine transformations (rotation, scaling, translation).
    """

    METADATA: Dict[str, Any] = {
        "id": "affine",
        "name": "Affine Transform",
        "order": 1,
        "params": {
            "rotation": {"type": "float", "min": -180.0, "max": 180.0, "default": 0.0},
            "scale": {"type": "float", "min": 0.5, "max": 1.5, "default": 1.0},
            "translate_x": {"type": "float", "min": -0.2, "max": 0.2, "default": 0.0},
            "translate_y": {"type": "float", "min": -0.2, "max": 0.2, "default": 0.0},
            "random": {"type": "bool", "default": True},
        },
    }

    rotation_range: List[float]
    scaling_range: List[float]
    translation: Optional[List[float]]
    transform: v2.RandomAffine

    def __init__(
        self,
        rotation_range: List[float] = [-30.0, 30.0],
        scaling_range: List[float] = [0.8, 1.2],
        translation: Optional[List[float]] = None
    ) -> None:
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.translation = translation
        self.transform = v2.RandomAffine(
            degrees=rotation_range,
            scale=tuple(scaling_range),  # type: ignore
            translate=translation,
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[Union[Image.Image, torch.Tensor], Optional[Any]]:
        is_random = bool(kwargs.get("random", True))

        if is_random:
            if joints is not None:
                return self.transform(image, joints)
            return self.transform(image), None

        # Fixed parameters
        rotation = float(kwargs.get("rotation", 0.0))
        scale = float(kwargs.get("scale", 1.0))
        tx = float(kwargs.get("translate_x", 0.0))
        ty = float(kwargs.get("translate_y", 0.0))

        if hasattr(image, "width"):
            img_pil = cast(Image.Image, image)
            img_w, img_h = img_pil.width, img_pil.height
        else:
            img_tensor = cast(torch.Tensor, image)
            img_w, img_h = img_tensor.shape[-1], img_tensor.shape[-2]

        translations = (tx * img_w, ty * img_h)

        if joints is not None:
            image = v2.functional.affine(
                image, angle=rotation, translate=translations, scale=scale, shear=[0.0, 0.0]
            )
            joints = v2.functional.affine(
                joints,
                angle=rotation,
                translate=translations,
                scale=scale,
                shear=[0.0, 0.0],
            )
            return image, joints

        return (
            v2.functional.affine(
                image, angle=rotation, translate=translations, scale=scale, shear=[0.0, 0.0]
            ),
            None,
        )
