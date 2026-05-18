from typing import Any, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import random
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


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
                    if j_np[2, pair[0]] < 2 and j_np[2, pair[1]] < 2:
                        coverage_options.append(min(j_np[1, pair[0]], j_np[1, pair[1]]))
            elif len(joints.shape) == 3:  # (1, 14, 2)
                j_np = joints[0].cpu().numpy()
                for pair in [(0, 5), (1, 4), (2, 3), (8, 9)]:
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
            # Convert back to float32 tensor normalized to [0, 1] to match standard pipeline
            return v2.functional.to_image(final_image).float().to(device) / 255.0
        return final_image


class ThermalIntensityJitter:
    """
    Randomly scales brightness and compresses/stretches contrast of thermal images
    to simulate dynamic range attenuation (e.g., from blankets) and subject/ambient shifts.
    """
    def __init__(self, probability: float = 0.5, intensity_range: tuple = (0.55, 1.15), contrast_range: tuple = (0.5, 1.15)):
        self.probability = probability
        self.intensity_range = intensity_range
        self.contrast_range = contrast_range

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_np = image.cpu().numpy().copy().astype(np.float32)
        else:
            img_np = np.array(image).copy().astype(np.float32)

        # 1. Multiplicative Brightness Scale
        brightness_scale = random.uniform(*self.intensity_range)
        img_np = img_np * brightness_scale

        # 2. Contrast Compression/Expansion
        mean_val = np.mean(img_np)
        contrast_scale = random.uniform(*self.contrast_range)
        img_np = (img_np - mean_val) * contrast_scale + mean_val

        # Clip values to valid image range
        max_val = 1.0 if is_tensor else 255.0
        img_np = np.clip(img_np, 0.0, max_val)

        if is_tensor:
            return torch.from_numpy(img_np).to(device)
        else:
            return Image.fromarray(img_np.astype(np.uint8))


class IRSensorNoise:
    """
    Injects simulated Gaussian readout noise and random dead/hot pixels to represent
    IR sensor limitations and regularize the network.
    """
    def __init__(self, probability: float = 0.4, noise_sigma: tuple = (5, 12), dead_pixel_ratio: float = 0.003):
        self.probability = probability
        self.noise_sigma = noise_sigma
        self.dead_pixel_ratio = dead_pixel_ratio

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_np = image.cpu().numpy().copy().astype(np.float32)
        else:
            img_np = np.array(image).copy().astype(np.float32)

        # 1. Add Gaussian Noise
        sigma = random.uniform(*self.noise_sigma)
        # Scale sigma if tensor [0, 1] instead of uint8 [0, 255]
        if is_tensor:
            sigma = sigma / 255.0
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        img_np = img_np + noise

        # 2. Add Salt & Pepper Dead/Hot Pixels
        num_pixels = int(self.dead_pixel_ratio * img_np.size)
        if num_pixels > 0:
            # Flatten indices
            flat_indices = np.random.choice(img_np.size, num_pixels, replace=False)
            unflatten_coords = np.unravel_index(flat_indices, img_np.shape)
            
            # 50% dead (0), 50% hot (max)
            dead_mask = np.random.random(num_pixels) < 0.5
            max_val = 1.0 if is_tensor else 255.0
            
            # We can index directly with coordinates
            img_np[unflatten_coords] = np.where(dead_mask, 0.0, max_val)

        # Clip values
        max_val = 1.0 if is_tensor else 255.0
        img_np = np.clip(img_np, 0.0, max_val)

        if is_tensor:
            return torch.from_numpy(img_np).to(device)
        else:
            return Image.fromarray(img_np.astype(np.uint8))


class CutoutAugmentation:
    """
    Zeroes out a randomly placed rectangular region in the image to simulate full occlusion.
    """
    def __init__(self, probability: float = 0.5, size_ratio: float = 0.35):
        self.probability = probability
        self.size_ratio = size_ratio

    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if random.random() > self.probability:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_np = image.cpu().numpy().copy()
            h, w = img_np.shape[-2], img_np.shape[-1]
        else:
            img_np = np.array(image).copy()
            h, w = img_np.shape[0], img_np.shape[1]

        # Determine cutout size randomly within [0.2, size_ratio] of image dimensions
        cutout_h = int(h * random.uniform(0.2, self.size_ratio))
        cutout_w = int(w * random.uniform(0.2, self.size_ratio))

        # Randomly choose top-left corner
        y0 = random.randint(0, h - cutout_h)
        x0 = random.randint(0, w - cutout_w)

        # Apply zero fill value (blackout)
        if is_tensor:
            if len(img_np.shape) == 3:  # (C, H, W)
                img_np[:, y0:y0+cutout_h, x0:x0+cutout_w] = 0.0
            else:
                img_np[y0:y0+cutout_h, x0:x0+cutout_w] = 0.0
        else:
            if len(img_np.shape) == 3:
                img_np[y0:y0+cutout_h, x0:x0+cutout_w, :] = 0
            else:
                img_np[y0:y0+cutout_h, x0:x0+cutout_w] = 0

        if is_tensor:
            return torch.from_numpy(img_np).to(device)
        else:
            return Image.fromarray(img_np)


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

        # Check for new robust augmentations in config
        self.cutout_prob = self.config.get("cutout_prob", 0.0)
        self.cutout_size_ratio = self.config.get("cutout_size_ratio", 0.35)
        
        self.intensity_jitter_prob = self.config.get("intensity_jitter_prob", 0.0)
        self.intensity_jitter_range = self.config.get("intensity_jitter_range", [0.55, 1.15])
        self.contrast_jitter_range = self.config.get("contrast_jitter_range", [0.5, 1.15])
        
        self.sensor_noise_prob = self.config.get("sensor_noise_prob", 0.0)
        self.sensor_noise_sigma = self.config.get("sensor_noise_sigma", [5, 12])

        # Instantiate new augmentations
        self.cutout_augmenter = CutoutAugmentation(
            probability=self.cutout_prob,
            size_ratio=self.cutout_size_ratio
        ) if self.cutout_prob > 0 else None

        self.intensity_jitter = ThermalIntensityJitter(
            probability=self.intensity_jitter_prob,
            intensity_range=tuple(self.intensity_jitter_range),
            contrast_range=tuple(self.contrast_jitter_range)
        ) if self.intensity_jitter_prob > 0 else None

        self.sensor_noise = IRSensorNoise(
            probability=self.sensor_noise_prob,
            noise_sigma=tuple(self.sensor_noise_sigma)
        ) if self.sensor_noise_prob > 0 else None

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
                    # Reorder joints for symmetry
                    flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                    vis = vis[flip_indices]

        # 3. Apply Affine (Rotation + Scaling + Translation)
        if self.affine_transform:
            if kpts is not None:
                image, kpts = self.affine_transform(image, kpts)
            else:
                image = self.affine_transform(image)

        # 4. Thermal diffusion
        source_image = image
        if return_pair:
            source_image = image.clone() if torch.is_tensor(image) else image.copy()

        if self.thermal_augmenter:
            image = self.thermal_augmenter(image, joints=kpts, is_ir=is_ir)

        # 4.5. New photometric and cutout augmentations (only if training and is_ir)
        if self.is_training and is_ir:
            if self.intensity_jitter:
                image = self.intensity_jitter(image)
            if self.sensor_noise:
                image = self.sensor_noise(image)
            if self.cutout_augmenter:
                image = self.cutout_augmenter(image)

        # 5. Final Assembly
        if kpts is not None:
            final_coords = kpts.view(14, 2).T  # (2, 14)
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = None

        if return_pair:
            return image, source_image, final_joints
        return image, final_joints


# Legacy alias
DataAugmenterV2 = DataAugmenter

