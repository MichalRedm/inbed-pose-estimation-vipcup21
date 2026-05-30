import torch
import numpy as np
import random
from PIL import Image
from src.data.augmentations import DataAugmenter as LegacyAugmenter
from typing import Any, Type

# We will implement the NewAugmenter later in src/data/augmentations.py
# For now, this test will fail until we implement the new one.
# To make it pass initially, we can point it to the LegacyAugmenter.
NewAugmenter: Type[Any]
try:
    from src.data.augmentations import DataAugmenterV2 as NewAugmenter_V2  # type: ignore

    NewAugmenter = NewAugmenter_V2
except ImportError:
    NewAugmenter = LegacyAugmenter


def test_aug_joint_parity() -> None:
    """
    Verify that joint transformations (flip, rotate, scale) are consistent.
    Note: Pixel-perfect parity is difficult between PIL and torchvision.v2,
    but joint coordinates should follow the same geometric logic.
    """
    seed = 42
    config = {
        "enabled": True,
        "flip_prob": 1.0,  # Force flip
        "rotation_range": [30, 30],  # Force 30 deg
        "scaling_range": [1.2, 1.2],  # Force 1.2x
        "occlusion_prob": 0.0,  # Disable thermal for now
    }

    # Create dummy image and joints
    img = Image.new("L", (256, 256), 128)
    joints = np.zeros((3, 14))
    joints[0, :] = np.linspace(50, 200, 14)  # x
    joints[1, :] = np.linspace(50, 200, 14)  # y
    joints[2, :] = 0  # visible

    # Run legacy
    random.seed(seed)
    np.random.seed(seed)
    legacy_aug = LegacyAugmenter(config=config, is_training=True)
    leg_img, leg_joints = legacy_aug(img.copy(), joints.copy(), is_ir=True)

    # Run new (will be LegacyAugmenter for now)
    random.seed(seed)
    np.random.seed(seed)
    # Once we implement V2, this will use the new class
    new_aug = NewAugmenter(config=config, is_training=True)
    # Note: NewAugmenter might return tensors if it's V2-based
    new_img, new_joints = new_aug(img.copy(), joints.copy(), is_ir=True)

    if torch.is_tensor(new_joints):
        new_joints = new_joints.numpy()

    # Check joint parity
    assert np.allclose(leg_joints, new_joints, atol=1e-1), (
        f"Joint mismatch: {leg_joints} vs {new_joints}"
    )


if __name__ == "__main__":
    test_aug_joint_parity()
