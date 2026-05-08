from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random


class ThermalDiffusionAugmenter:
    """
    Simulates the effect of a blanket on IR images by diffusing and dampening 
    the heat signature. It uses joint coordinates to realistically place 
    the "blanket" over the subject.
    """

    def __init__(self, probability: float = 0.5, is_training: bool = True):
        self.probability = probability
        self.is_training = is_training

    def __call__(self, image: Image.Image, joints: Optional[np.ndarray], is_ir: bool) -> Image.Image:
        # Skip if not training, if it's NOT an IR image, or if the random check fails
        if not self.is_training or not is_ir or random.random() > self.probability:
            return image

        w, h = image.size
        
        # Create a mask for the "covered" area
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Determine coverage level (e.g., from ankles to knees, hips, or chest)
        # LSP indices: 0,5: ankles | 1,4: knees | 2,3: hips | 8,9: shoulders
        coverage_options = []
        if joints is not None:
            # Check visibility/annotation
            if joints[2, 0] < 2 and joints[2, 5] < 2: # ankles
                coverage_options.append(min(joints[1, 0], joints[1, 5]))
            if joints[2, 1] < 2 and joints[2, 4] < 2: # knees
                coverage_options.append(min(joints[1, 1], joints[1, 4]))
            if joints[2, 2] < 2 and joints[2, 3] < 2: # hips
                coverage_options.append(min(joints[1, 2], joints[1, 3]))
            if joints[2, 8] < 2 and joints[2, 9] < 2: # shoulders
                coverage_options.append(min(joints[1, 8], joints[1, 9]))

        if coverage_options:
            # Pick a random joint level to start the blanket from
            base_y = random.choice(coverage_options)
        else:
            # Fallback to random height if no joints are visible
            base_y = random.randint(int(h * 0.2), int(h * 0.6))

        # Add slight randomization to the top edge
        left_y = base_y + random.randint(-20, 20)
        right_y = base_y + random.randint(-20, 20)
        
        polygon = [(0, left_y), (w, right_y), (w, h), (0, h)]
        draw.polygon(polygon, fill=255)

        # Soften the mask edge
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

        # 1. Create a blurred version of the image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 7)))

        # 2. Create a dampened version (lower intensity)
        # Convert to numpy for intensity manipulation
        img_np = np.array(image).astype(np.float32)
        dampened_np = img_np * random.uniform(0.5, 0.8)
        
        # Add slight thermal noise
        noise = np.random.normal(0, 3, dampened_np.shape).astype(np.float32)
        dampened_np = np.clip(dampened_np + noise, 0, 255).astype(np.uint8)
        dampened = Image.fromarray(dampened_np)

        # 3. Combine blur and dampening
        # We blend the original with the blurred+dampened version using the mask
        blurred_dampened = Image.composite(dampened.filter(ImageFilter.GaussianBlur(radius=2)), blurred, mask)
        
        # Final composition: original image blended with the "under-blanket" version
        final_image = Image.composite(blurred_dampened, image, mask)

        return final_image


class DataAugmenter:
    """
    Modular data augmentation for pose estimation.
    Handles simultaneous transformation of images and joint coordinates.
    """

    def __init__(self, config: Optional[dict] = None, is_training: bool = True):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.is_training = is_training

        # Initialize the thermal diffusion augmenter with config parameters
        self.thermal_augmenter = ThermalDiffusionAugmenter(
            probability=self.config.get("occlusion_prob", 0.5),
            is_training=self.is_training,
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
        image, joints = self._random_flip(image, joints)
        image, joints = self._random_rotate(image, joints)
        image, joints = self._random_scale(image, joints)

        # 2. Pixel-level transforms (affects only image, joints stay the same)
        # image, joints = self._color_jitter(image, joints)

        # 3. Apply the IR-only thermal diffusion (simulates blanket)
        if self.thermal_augmenter:
            image = self.thermal_augmenter(image, joints=joints, is_ir=is_ir)

        return image, joints

    def _random_flip(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Horizontal flip with probability flip_prob."""
        flip_prob = self.config.get("flip_prob", 0.5)
        if random.random() > flip_prob:
            return image, joints

        # Flip image
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Flip joints
        if joints is not None:
            # joints: (3, 14) -> (x, y, vis)
            # x' = width - 1 - x
            w, _ = image.size
            joints = joints.copy()
            joints[0] = w - 1 - joints[0]

            # Reorder joints for symmetry (left side becomes right side)
            # LSP joint order: 0: R ankle, 1: R knee, 2: R hip, 3: L hip, 4: L knee, 5: L ankle,
            # 6: R wrist, 7: R elbow, 8: R shoulder, 9: L shoulder, 10: L elbow, 11: L wrist,
            # 12: neck, 13: head
            flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
            joints = joints[:, flip_indices]

        return image, joints

    def _random_rotate(
        self, image: Image.Image, joints: Any
    ) -> Tuple[Image.Image, Any]:
        """Random rotation within rotation_range."""
        rot_range = self.config.get("rotation_range", [-30, 30])
        angle = random.uniform(rot_range[0], rot_range[1])

        if angle == 0:
            return image, joints

        # Rotate image
        # Using BICUBIC for better quality, but IR might prefer NEAREST/BILINEAR
        # We rotate around center
        w, h = image.size
        center = (w / 2, h / 2)
        image = image.rotate(angle, resample=Image.BILINEAR)

        # Rotate joints
        if joints is not None:
            joints = joints.copy()
            angle_rad = -np.deg2rad(
                angle
            )  # PIL rotate is counter-clockwise, math is usually clockwise
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Shift to origin
            x = joints[0] - center[0]
            y = joints[1] - center[1]

            # Rotate
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a

            # Shift back
            joints[0] = new_x + center[0]
            joints[1] = new_y + center[1]

        return image, joints

    def _random_scale(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Random scaling within scaling_range."""
        scale_range = self.config.get("scaling_range", [0.8, 1.2])
        scale = random.uniform(scale_range[0], scale_range[1])

        if scale == 1.0:
            return image, joints

        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        image_resized = image.resize((new_w, new_h), resample=Image.BILINEAR)

        # Center crop or pad to original size
        image_final = Image.new(image.mode, (w, h), color=0)

        # Calculate offsets
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2

        # Paste onto new background
        if scale < 1.0:
            # Smaller: paste in center
            image_final.paste(image_resized, (offset_x, offset_y))
        else:
            # Larger: crop center
            image_final = image_resized.crop(
                (-offset_x, -offset_y, -offset_x + w, -offset_y + h)
            )

        # Update joints
        if joints is not None:
            joints = joints.copy()
            # Scale
            joints[0] *= scale
            joints[1] *= scale
            # Offset
            joints[0] += offset_x
            joints[1] += offset_y

        return image_final, joints

    def _color_jitter(self, image: Image.Image, joints: Any) -> Tuple[Image.Image, Any]:
        """Placeholder for color augmentation (image only)."""
        return image, joints
