from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

class OcclusionAugmenter:
    """
    Adds a mask that imitates a bed sheet covering the patient.
    It covers the lower portion of the image with a semi-transparent overlay.
    Applied ONLY to IR images.
    """

    def __init__(self, probability: float = 0.5, is_training: bool = True):
        self.probability = probability
        self.is_training = is_training

    def __call__(self, image: Image.Image, is_ir: bool) -> Image.Image:
        # Skip if not training, if it's NOT an IR image, or if the random check fails
        if not self.is_training or not is_ir or random.random() > self.probability:
            return image

        w, h = image.size
        
        # Create a transparent overlay for the "sheet"
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Determine the top edge of the sheet (between 20% and 50% of the height)
        base_y = random.randint(int(h * 0.2), int(h * 0.5))
        
        # Add slight randomization to the left and right sides to simulate an angled/uneven sheet
        left_y = base_y + random.randint(-int(h * 0.05), int(h * 0.05))
        right_y = base_y + random.randint(-int(h * 0.05), int(h * 0.05))
        
        # Define the polygon for the bedsheet (bottom part of the image)
        polygon = [(0, left_y), (w, right_y), (w, h), (0, h)]
        
        # Sheet color for IR: Ambient room temperature is usually cooler (darker).
        # We use a dark gray/black with high opacity to block most of the "heat" 
        # but let a tiny bit bleed through, which is realistic for thin hospital sheets.
        sheet_color = (30, 30, 30, 230) 
        draw.polygon(polygon, fill=sheet_color)
        
        # Apply a Gaussian blur to the overlay to soften the edge of the sheet
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Composite the original image with the blurred sheet overlay
        original_mode = image.mode
        image_rgba = image.convert('RGBA')
        blended = Image.alpha_composite(image_rgba, overlay)
        
        # Return the image in its original format (e.g., 'RGB' or 'L')
        return blended.convert(original_mode)


class DataAugmenter:
    """
    Modular data augmentation for pose estimation.
    Handles simultaneous transformation of images and joint coordinates.
    """

    def __init__(self, config: Optional[dict] = None, is_training: bool = True):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.is_training = is_training
        
        # Initialize the occlusion augmenter with config parameters
        self.occlusion_augmenter = OcclusionAugmenter(
            probability=self.config.get("occlusion_prob", 0.5),
            is_training=self.is_training
        )

    def __call__(
        self, image: Image.Image, joints: Optional[np.ndarray], is_ir: bool = False
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Apply enabled augmentations.
        image: PIL Image
        joints: numpy array of shape (3, 14) -> (x, y, visibility)
        is_ir: boolean flag indicating if the current image is an Infrared/Thermal modality
        returns: (transformed_image, transformed_joints)
        """
        if not self.enabled:
            return image, joints

        # 1. Spatial/Geometric transforms (would affect both image and joints)
        # image, joints = self._random_flip(image, joints)
        # image, joints = self._random_rotate(image, joints)
        # image, joints = self._random_scale(image, joints)

        # 2. Pixel-level transforms (affects only image, joints stay the same)
        # image, joints = self._color_jitter(image, joints)

        # 3. Apply the IR-only sheet occlusion
        if self.occlusion_augmenter:
            image = self.occlusion_augmenter(image, is_ir=is_ir)

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