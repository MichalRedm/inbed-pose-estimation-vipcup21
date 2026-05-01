from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image


class DataAugmenter:
    """
    Modular data augmentation for pose estimation.
    Handles simultaneous transformation of images and joint coordinates.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)

    def __call__(
        self, image: Image.Image, joints: Optional[np.ndarray]
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Apply enabled augmentations.
        image: PIL Image
        joints: numpy array of shape (3, 14) -> (x, y, visibility)
        returns: (transformed_image, transformed_joints)
        """
        if not self.enabled:
            return image, joints

        # TODO: Implement augmentations here
        # Example placeholders for future additions:
        # image, joints = self._random_flip(image, joints)
        # image, joints = self._random_rotate(image, joints)
        # image, joints = self._random_scale(image, joints)
        # image, joints = self._color_jitter(image, joints)

        return image, joints

    def _random_flip(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Placeholder for horizontal flip."""
        return image, joints

    def _random_rotate(
        self, image: Image.Image, joints: Any
    ) -> Tuple[Image.Image, Any]:
        """Placeholder for random rotation."""
        return image, joints

    def _random_scale(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Placeholder for random scaling and cropping."""
        return image, joints

    def _color_jitter(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Placeholder for color augmentation (image only)."""
        return image, joints
