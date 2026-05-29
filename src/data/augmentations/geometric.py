import random
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors

class HorizontalFlipAugmentation:
    """
    Randomly flips the image horizontally and reorders the joints for symmetry.
    """

    METADATA = {
        "id": "flip",
        "name": "Horizontal Flip",
        "order": 0,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "force_apply": {"type": "bool", "default": False},
        },
    }

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image, joints=None, **kwargs):
        prob = kwargs.get("probability", self.probability)
        force = kwargs.get("force_apply", False)

        if not force and random.random() > prob:
            return image, joints

        # 1. Image Flip
        image = v2.functional.hflip(image)

        # 2. Joints Flip & Reorder
        if joints is not None:
            # Handle tv_tensors.KeyPoints
            if torch.is_tensor(joints) and hasattr(joints, "canvas_size"):
                kpts = v2.functional.hflip(joints)
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                kpts = tv_tensors.KeyPoints(
                    kpts[:, flip_indices, :], canvas_size=kpts.canvas_size
                )
                return image, kpts

            # Manual reorder for numpy/raw tensor (3, 14)
            img_w = image.width if hasattr(image, "width") else image.shape[-1]
            joints = joints.copy()
            joints[0, :] = img_w - joints[0, :]  # Flip X
            flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
            joints = joints[:, flip_indices]

        return image, joints


class AffineAugmentation:
    """
    Applies random affine transformations (rotation, scaling, translation).
    """

    METADATA = {
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

    def __init__(
        self, rotation_range=[-30, 30], scaling_range=[0.8, 1.2], translation=None
    ):
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.translation = translation
        self.transform = v2.RandomAffine(
            degrees=rotation_range,
            scale=scaling_range,
            translate=translation,
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def __call__(self, image, joints=None, **kwargs):
        is_random = kwargs.get("random", True)

        if is_random:
            if joints is not None:
                return self.transform(image, joints)
            return self.transform(image), None

        # Fixed parameters
        rotation = kwargs.get("rotation", 0.0)
        scale = kwargs.get("scale", 1.0)
        tx = kwargs.get("translate_x", 0.0)
        ty = kwargs.get("translate_y", 0.0)

        img_w, img_h = (
            (image.width, image.height)
            if hasattr(image, "width")
            else (image.shape[-1], image.shape[-2])
        )
        translations = (tx * img_w, ty * img_h)

        if joints is not None:
            image = v2.functional.affine(
                image, angle=rotation, translate=translations, scale=scale, shear=[0, 0]
            )
            joints = v2.functional.affine(
                joints,
                angle=rotation,
                translate=translations,
                scale=scale,
                shear=[0, 0],
            )
            return image, joints

        return (
            v2.functional.affine(
                image, angle=rotation, translate=translations, scale=scale, shear=[0, 0]
            ),
            None,
        )
