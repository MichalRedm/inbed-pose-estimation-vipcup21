from typing import Any, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import random
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


class CutoutAugmentation:
    """
    Zeros out a randomly placed rectangular region of size up to ratio*w, ratio*h.
    Directly targets model occlusion robustness and context-learning.
    """

    def __init__(self, probability: float = 0.5, size_ratio: float = 0.35):
        self.probability = probability
        self.size_ratio = size_ratio

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        w, h = img_pil.size

        # Generate random box size up to size_ratio
        box_w = random.randint(int(w * 0.15), int(w * self.size_ratio))
        box_h = random.randint(int(h * 0.15), int(h * self.size_ratio))

        # Random location
        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)

        # Draw a black rectangle (0) on a copy of the image
        img_pil = img_pil.copy()
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=0)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class ThermalIntensityJitter:
    """
    Randomly dampens or slightly boosts contrast and brightness to simulate thermal attenuation.
    Directly addresses the 53% dynamic range gap measured in dataset analysis.
    """

    def __init__(
        self,
        probability: float = 0.5,
        brightness_range: list[float] = [0.55, 1.15],
        contrast_range: list[float] = [0.5, 1.15],
    ):
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        # Brightness jitter: uniform scaling
        scale_b = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img_np = img_np * scale_b

        # Contrast jitter: stretch/compress around mean
        scale_c = random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = img_np.mean()
        img_np = (img_np - mean) * scale_c + mean

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class IRSensorNoise:
    """
    Simulates readout thermal/Gaussian noise and dead/hot pixels (salt & pepper).
    Regularizes against spatial texture over-reliance.
    """

    def __init__(
        self,
        probability: float = 0.4,
        sigma_range: list[float] = [5.0, 12.0],
        sp_prob: float = 0.003,
    ):
        self.probability = probability
        self.sigma_range = sigma_range
        self.sp_prob = sp_prob

    def __call__(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        # 1. Gaussian noise
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        img_np = img_np + noise

        # 2. Salt & Pepper noise (dead/hot pixels)
        sp_mask = np.random.random(img_np.shape[:2])
        # Salt (hot pixels) -> 255
        img_np[sp_mask < (self.sp_prob / 2.0)] = 255.0
        # Pepper (dead pixels) -> 0
        img_np[sp_mask > (1.0 - self.sp_prob / 2.0)] = 0.0

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class ThermalDiffusionAugmenter:
    """
    Simulates the effect of a blanket on IR images by diffusing and dampening
    the heat signature. It uses joint coordinates to realistically place
    the "blanket" over the subject.
    """

    def __init__(self, probability: float = 0.5, is_training: bool = True):
        self.probability = probability
        self.is_training = is_training

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[torch.Tensor],
        is_ir: bool,
    ) -> Union[Image.Image, torch.Tensor]:
        # Skip if not training, if it's NOT an IR image, or if the random check fails
        if not self.is_training or not is_ir or random.random() > self.probability:
            return image

        # Handle both PIL and Tensor
        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        w, h = img_pil.size

        # Determine coverage level
        coverage_options = []
        if joints is not None:
            # joints: (1, 14, 2) or (3, 14) or (14, 2)
            if joints.shape[0] == 3:
                j_np = joints.cpu().numpy()
                for pair in [(0, 5), (1, 4), (2, 3), (8, 9)]:
                    if pair[0] < j_np.shape[1] and pair[1] < j_np.shape[1]:
                        if j_np[2, pair[0]] < 2 and j_np[2, pair[1]] < 2:
                            coverage_options.append(
                                min(j_np[1, pair[0]], j_np[1, pair[1]])
                            )
            elif len(joints.shape) == 3:  # (1, N, 2)
                j_np = joints[0].cpu().numpy()
                for pair in [(0, 5), (1, 4), (2, 3), (8, 9)]:
                    if pair[0] < j_np.shape[0] and pair[1] < j_np.shape[0]:
                        coverage_options.append(min(j_np[pair[0], 1], j_np[pair[1], 1]))

        full_coverage = random.random() < 0.1
        if full_coverage:
            base_y = 0
        elif coverage_options:
            base_y = random.choice(coverage_options)
        else:
            base_y = random.randint(int(h * 0.1), int(h * 0.7))

        # Create wavy mask
        mask = Image.new("L", img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        points = []
        num_points = 20
        freq = random.uniform(2, 5)
        amp = random.uniform(5, 20)
        phase = random.uniform(0, 2 * np.pi)

        for i in range(num_points + 1):
            x = int(i * w / num_points)
            y = int(base_y + amp * np.sin(freq * (x / w) * 2 * np.pi + phase))
            y = max(0, min(h - 1, y))
            points.append((x, y))

        points.append((w, h))
        points.append((0, h))
        draw.polygon(points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(5, 10)))

        # 1. Blur
        blur_radius = random.uniform(4, 10)
        blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 2. Dampen & Noise
        img_np = np.array(img_pil).astype(np.float32)
        damp_factor = random.uniform(0.4, 0.7)
        dampened_np = img_np * damp_factor
        noise = np.random.normal(0, 5, dampened_np.shape).astype(np.float32)
        dampened_np = np.clip(dampened_np + noise, 0, 255)

        if random.random() < 0.3:
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            texture = 5 * np.sin(grid_x / 5) * np.cos(grid_y / 5)
            dampened_np = np.clip(dampened_np + texture, 0, 255)

        dampened = Image.fromarray(dampened_np.astype(np.uint8))

        # 3. Combine
        blanket_layer = Image.composite(
            dampened.filter(ImageFilter.GaussianBlur(radius=2)), blurred, mask
        )
        final_image = Image.composite(blanket_layer, img_pil, mask)

        if is_tensor:
            return v2.functional.to_image(final_image).to(device)
        return final_image


class DataAugmenter:
    """
    Optimized Data Augmentation using torchvision.transforms.v2.
    """

    def __init__(self, config: Optional[dict] = None, is_training: bool = True):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.is_training = is_training

        # Thermal augmenter
        self.thermal_augmenter = ThermalDiffusionAugmenter(
            probability=self.config.get("occlusion_prob", 0.5),
            is_training=self.is_training,
        )

        # New augmentations setup
        self.intensity_jitter = None
        if (
            self.enabled
            and self.is_training
            and self.config.get("intensity_jitter_prob", 0.0) > 0
        ):
            self.intensity_jitter = ThermalIntensityJitter(
                probability=self.config.get("intensity_jitter_prob", 0.5),
                brightness_range=self.config.get(
                    "intensity_jitter_range", [0.55, 1.15]
                ),
                contrast_range=self.config.get("contrast_jitter_range", [0.5, 1.15]),
            )

        self.sensor_noise = None
        if (
            self.enabled
            and self.is_training
            and self.config.get("sensor_noise_prob", 0.0) > 0
        ):
            self.sensor_noise = IRSensorNoise(
                probability=self.config.get("sensor_noise_prob", 0.4),
                sigma_range=self.config.get("sensor_noise_sigma", [5.0, 12.0]),
            )

        self.cutout = None
        if (
            self.enabled
            and self.is_training
            and self.config.get("cutout_prob", 0.0) > 0
        ):
            self.cutout = CutoutAugmentation(
                probability=self.config.get("cutout_prob", 0.5),
                size_ratio=self.config.get("cutout_size_ratio", 0.35),
            )

        # Spatial transforms (Affine only, Flip handled manually)
        if self.enabled and self.is_training:
            rot_range = self.config.get("rotation_range", [-30, 30])
            scale_range = self.config.get("scaling_range", [0.8, 1.2])
            translate = self.config.get("translation", None)
            if translate is not None:
                translate = tuple(translate)

            self.affine_transform = v2.RandomAffine(
                degrees=rot_range,
                scale=scale_range,
                translate=translate,
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            self.affine_transform = None

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[np.ndarray],
        is_ir: bool = False,
        return_pair: bool = False,
    ) -> Any:
        if not self.enabled:
            if return_pair:
                return image, image, joints
            return image, joints

        # 1. Prepare Keypoints
        if joints is not None:
            coords = torch.from_numpy(joints[:2, :].T).float().unsqueeze(0)
            vis = torch.from_numpy(joints[2, :]).float()
            w, h = (
                (image.width, image.height)
                if hasattr(image, "width")
                else (image.shape[-1], image.shape[-2])
            )
            kpts = tv_tensors.KeyPoints(coords, canvas_size=(h, w))
        else:
            kpts = None
            vis = None

        # 2. Manual Horizontal Flip (for joint reordering)
        if self.is_training:
            flip_prob = self.config.get("flip_prob", 0.5)
            if random.random() < flip_prob:
                image = v2.functional.hflip(image)
                if kpts is not None:
                    kpts = v2.functional.hflip(kpts)
                    # Reorder joints for symmetry (must reorder BOTH keypoint coordinates and visibility)
                    flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                    kpts = tv_tensors.KeyPoints(
                        kpts[:, flip_indices, :], canvas_size=kpts.canvas_size
                    )
                    vis = vis[flip_indices]

        # 3. Apply Affine (Rotation + Scaling + Translation)
        if self.affine_transform:
            if kpts is not None:
                image, kpts = self.affine_transform(image, kpts)
            else:
                image = self.affine_transform(image)

        # 4. New Appearance Augmentations (Intensity jitter & noise)
        if self.intensity_jitter:
            image = self.intensity_jitter(image)

        if self.sensor_noise:
            image = self.sensor_noise(image)

        # 5. Thermal diffusion (blanket simulation)
        source_image = image
        if return_pair:
            source_image = image.clone() if torch.is_tensor(image) else image.copy()

        if self.thermal_augmenter:
            image = self.thermal_augmenter(image, joints=kpts, is_ir=is_ir)

        # 6. Structured Cutout (post-blanket occlusion simulation)
        if self.cutout:
            image = self.cutout(image)

        # 7. Final Assembly
        if kpts is not None:
            num_kpts = kpts.shape[1]
            final_coords = kpts.view(num_kpts, 2).T  # (2, num_kpts)
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = None

        if return_pair:
            return image, source_image, final_joints
        return image, final_joints


# Legacy alias
DataAugmenterV2 = DataAugmenter
