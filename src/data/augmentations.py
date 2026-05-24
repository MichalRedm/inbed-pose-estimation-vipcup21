from typing import Any, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import random
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
import inspect
import sys

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
            if torch.is_tensor(joints) and hasattr(joints, "canvas_size"):
                kpts = v2.functional.hflip(joints)
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                # Guard against structures that don't match the 14-keypoint layout
                if kpts.shape[1] == len(flip_indices):
                    kpts = tv_tensors.KeyPoints(
                        kpts[:, flip_indices, :], canvas_size=kpts.canvas_size
                    )
                return image, kpts

            # Manual fallback handling safely for arrays shaped (3, 14) or (2, 14)
            img_w = image.width if hasattr(image, "width") else image.shape[-1]
            if isinstance(joints, np.ndarray):
                joints = joints.copy()
                joints[0, :] = img_w - joints[0, :]
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                if joints.shape[1] == len(flip_indices):
                    joints = joints[:, flip_indices]
            elif torch.is_tensor(joints):
                joints = joints.clone()
                joints[0, :] = img_w - joints[0, :]
                flip_indices = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
                if joints.shape[1] == len(flip_indices):
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

    def __init__(self, rotation_range=[-30, 30], scaling_range=[0.8, 1.2], translation=None):
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

        rotation = kwargs.get("rotation", 0.0)
        scale = kwargs.get("scale", 1.0)
        tx = kwargs.get("translate_x", 0.0)
        ty = kwargs.get("translate_y", 0.0)

        img_w, img_h = (
            (image.width, image.height)
            if hasattr(image, "width")
            else (image.shape[-1], image.shape[-2])
        )
        translations = [int(tx * img_w), int(ty * img_h)]

        if joints is not None:
            image = v2.functional.affine(
                image, angle=rotation, translate=translations, scale=scale, shear=[0.0, 0.0]
            )
            joints = v2.functional.affine(
                joints, angle=rotation, translate=translations, scale=scale, shear=[0.0, 0.0]
            )
            return image, joints

        return v2.functional.affine(
            image, angle=rotation, translate=translations, scale=scale, shear=[0.0, 0.0]
        ), None


class ThermalIntensityJitter:
    """
    Randomly dampens or slightly boosts contrast and brightness to simulate thermal attenuation.
    """
    METADATA = {
        "id": "intensity_jitter",
        "name": "Thermal Intensity Jitter",
        "order": 2,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "brightness": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0},
            "contrast": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0},
        },
    }

    def __init__(self, probability: float = 0.5, brightness_range=[0.55, 1.15], contrast_range=[0.5, 1.15]):
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image: Union[Image.Image, torch.Tensor], **kwargs) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            # Handle standard single channel squeeze cases safely
            img_pil = v2.functional.to_pil_image(image.cpu())
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        scale_b = kwargs.get("brightness", random.uniform(self.brightness_range[0], self.brightness_range[1]))
        img_np = img_np * scale_b

        scale_c = kwargs.get("contrast", random.uniform(self.contrast_range[0], self.contrast_range[1]))
        mean = img_np.mean()
        img_np = (img_np - mean) * scale_c + mean

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class IRSensorNoise:
    """
    Simulates readout thermal/Gaussian noise and dead/hot pixels.
    """
    METADATA = {
        "id": "sensor_noise",
        "name": "IR Sensor Noise",
        "order": 3,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "sigma": {"type": "float", "min": 0.0, "max": 30.0, "default": 8.0},
            "sp_prob": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.003},
        },
    }

    def __init__(self, probability: float = 0.4, sigma_range=[5.0, 12.0], sp_prob: float = 0.003):
        self.probability = probability
        self.sigma_range = sigma_range
        self.sp_prob = sp_prob

    def __call__(self, image: Union[Image.Image, torch.Tensor], **kwargs) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image.cpu())
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        sigma = kwargs.get("sigma", random.uniform(self.sigma_range[0], self.sigma_range[1]))
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        img_np = img_np + noise

        sp_prob = kwargs.get("sp_prob", self.sp_prob)
        sp_mask = np.random.random(img_np.shape[:2])
        img_np[sp_mask < (sp_prob / 2.0)] = 255.0
        img_np[sp_mask > (1.0 - sp_prob / 2.0)] = 0.0

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class ThermalDiffusionAugmenter:
    """
    Simulates the effect of a blanket on IR images by diffusing and dampening the heat signature.
    """
    METADATA = {
        "id": "thermal_diffusion",
        "name": "Thermal Diffusion (Blanket)",
        "order": 4,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "base_y_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "damp_factor": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.55},
            "blur_radius": {"type": "float", "min": 0.0, "max": 20.0, "default": 7.0},
        },
    }

    def __init__(self, probability: float = 0.5, is_training: bool = True):
        self.probability = probability
        self.is_training = is_training

    def __call__(self, image: Union[Image.Image, torch.Tensor], joints: Optional[torch.Tensor] = None, is_ir: bool = True, **kwargs) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if not is_ir or random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image.cpu())
        else:
            img_pil = image

        original_mode = img_pil.mode
        if original_mode != "L":
            img_pil = img_pil.convert("L")

        w, h = img_pil.size
        head_y = None
        shoulders_y, hips_y, knees_y = [], [], []

        if joints is not None:
            if torch.is_tensor(joints):
                if len(joints.shape) == 3:
                    j_np = joints[0].cpu().numpy()
                elif len(joints.shape) == 2:
                    j_np = joints.get("array", joints.cpu().numpy()) if hasattr(joints, "array") else joints.cpu().numpy()
                    if j_np.shape[0] >= 3: 
                        j_np = j_np[:2, :].T
                else:
                    j_np = np.array(joints.cpu())
            else:
                j_np = np.array(joints)

            if len(j_np.shape) == 2 and j_np.shape[0] >= 14:
                if j_np[13, 0] > 0 or j_np[13, 1] > 0:
                    head_y = j_np[13, 1]
                for s_idx in [8, 9]:
                    if j_np[s_idx, 0] > 0 or j_np[s_idx, 1] > 0: shoulders_y.append(j_np[s_idx, 1])
                for h_idx in [2, 3]:
                    if j_np[h_idx, 0] > 0 or j_np[h_idx, 1] > 0: hips_y.append(j_np[h_idx, 1])
                for k_idx in [1, 4]:
                    if j_np[k_idx, 0] > 0 or j_np[k_idx, 1] > 0: knees_y.append(j_np[k_idx, 1])

        min_allowed_y = int(h * 0.15)
        if head_y is not None:
            min_allowed_y = max(min_allowed_y, int(head_y + 15))

        if "base_y_ratio" in kwargs:
            base_y = int(kwargs["base_y_ratio"] * h)
            base_y = max(min_allowed_y, base_y)
        else:
            coverage_choices = []
            if shoulders_y: coverage_choices.append(min(shoulders_y))
            if hips_y: coverage_choices.append(min(hips_y))
            if knees_y: coverage_choices.append(min(knees_y))

            if coverage_choices:
                base_y = random.choice(coverage_choices) + random.randint(-15, 15)
            else:
                base_y = random.randint(int(h * 0.25), int(h * 0.7))
            base_y = max(min_allowed_y, min(base_y, h - 20))

        base_y = int(base_y)

        # Wavy edge generation
        mask = Image.new("L", img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        wavy_points = []
        num_points = 20
        freq, amp, phase = random.uniform(1.5, 3.5), random.uniform(4, 15), random.uniform(0, 2 * np.pi)

        for i in range(num_points + 1):
            x = int(i * w / num_points)
            y = int(base_y + amp * np.sin(freq * (x / w) * 2 * np.pi + phase))
            wavy_points.append((x, max(0, min(h - 1, y))))

        polygon_points = list(wavy_points) + [(w, h), (0, h)]
        draw.polygon(polygon_points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 7)))

        img_np = np.array(img_pil).astype(np.float32)
        ambient_est = np.percentile(img_np, 15)

        # Generates folds & wrinkles
        drape_mask = Image.new("L", img_pil.size, 255)
        drape_draw = ImageDraw.Draw(drape_mask)
        for _ in range(random.randint(3, 6)):
            fx1, fy1 = random.randint(-40, w + 40), random.randint(base_y - 40, h + 40)
            fx2, fy2 = random.randint(-40, w + 40), fy1 + random.randint(80, 250)
            cx, cy = random.randint(min(fx1, fx2) - 50, max(fx1, fx2) + 50), (fy1 + fy2) // 2 + random.randint(-20, 20)
            pts = [
                (int((1 - t)**2 * fx1 + 2 * (1 - t) * t * cx + t**2 * fx2),
                 int((1 - t)**2 * fy1 + 2 * (1 - t) * t * cy + t**2 * fy2))
                for t in np.linspace(0, 1, 15)
            ]
            drape_draw.line(pts, fill=random.randint(40, 120), width=random.randint(25, 50), joint="round")

        drape_np = np.array(drape_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(10.0, 18.0)))).astype(np.float32) / 255.0
        combined_drape_np = drape_np

        blanket_ambient = ambient_est + (combined_drape_np - 0.85) * 12.0
        blanket_base = blanket_ambient + np.random.normal(0, 1.5, img_np.shape).astype(np.float32)

        body_heat = np.maximum(img_np - ambient_est, 0.0)
        max_heat = np.max(body_heat) + 1e-5
        body_heat_boosted = np.power(body_heat / max_heat, 1.3) * np.max(body_heat)
        heat_pil = Image.fromarray(np.clip(body_heat_boosted, 0, 255).astype(np.uint8))

        bloom_img = heat_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(8.0, 16.0)))
        contact_img = heat_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 5.0)))
        mixed_heat_np = np.array(bloom_img).astype(np.float32) * (1.0 - combined_drape_np) + np.array(contact_img).astype(np.float32) * combined_drape_np

        damp_factor = kwargs.get("damp_factor", random.uniform(0.18, 0.42))
        transmitted_heat = mixed_heat_np * damp_factor * combined_drape_np
        dampened_np = blanket_base + transmitted_heat

        # Crease Shadowing
        shadow_mask = Image.new("L", img_pil.size, 0)
        ImageDraw.Draw(shadow_mask).line(wavy_points, fill=255, width=random.randint(4, 8), joint="round")
        shadow_np = np.array(shadow_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))).astype(np.float32) / 255.0
        dampened_np *= (1.0 - shadow_np * random.uniform(0.03, 0.08))

        dampened = Image.fromarray(np.clip(dampened_np, 0, 255).astype(np.uint8))
        final_image = Image.composite(dampened, img_pil, mask)

        if original_mode != "L":
            final_image = final_image.convert(original_mode)

        if is_tensor:
            return v2.functional.to_image(final_image).to(device)
        return final_image


class SelfGhostingThermalFootprint:
    """
    Simulates a residual heat footprint by extracting the current body's heat,
    spatially shifting it, decaying it, and blurring it.
    """
    METADATA = {
        "id": "self_ghosting_footprint",
        "name": "Self-Ghosting Thermal Footprint",
        "order": 5,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "max_shift": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.15},
            "decay_factor": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.7},
            "blur_radius": {"type": "float", "min": 2.0, "max": 20.0, "default": 6.0},
        },
    }

    # MODIFIED: Higher decay range (brighter ghost) and lower blur (sharper ghost)
    def __init__(self, probability: float = 0.5, max_shift: float = 0.15, decay_range=[0.5, 0.85], blur_range=[4.0, 10.0]):
        self.probability = probability
        self.max_shift = max_shift
        self.decay_range = decay_range
        self.blur_range = blur_range

    def __call__(self, image: Union[Image.Image, torch.Tensor], **kwargs) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image.cpu())
        else:
            img_pil = image

        original_mode = img_pil.mode
        if original_mode != "L":
            img_pil = img_pil.convert("L")

        w, h = img_pil.size
        img_np = np.array(img_pil).astype(np.float32)

        ambient_est = np.percentile(img_np, 15)
        
        # MODIFIED: Multiply extracted heat by 1.5 to artificially boost the ghost intensity
        subject_heat = np.maximum(img_np - ambient_est, 0.0)
        subject_heat = subject_heat * 1.5 
        heat_pil = Image.fromarray(np.clip(subject_heat, 0, 255).astype(np.uint8))

        max_s = kwargs.get("max_shift", self.max_shift)
        max_shift_x, max_shift_y = int(w * max_s), int(h * max_s)
        shift_x = random.randint(-max_shift_x, max_shift_x) if max_shift_x > 0 else 0
        shift_y = random.randint(-max_shift_y, max_shift_y) if max_shift_y > 0 else 0
        rotation = random.uniform(-12.0, 12.0)

        shifted_heat = heat_pil.rotate(rotation, translate=(shift_x, shift_y))
        decay = kwargs.get("decay_factor", random.uniform(self.decay_range[0], self.decay_range[1]))
        blur_rad = kwargs.get("blur_radius", random.uniform(self.blur_range[0], self.blur_range[1]))

        shifted_heat = shifted_heat.filter(ImageFilter.GaussianBlur(radius=blur_rad))
        residual_np = np.array(shifted_heat).astype(np.float32) * decay
        combined_np = np.maximum(img_np, ambient_est + residual_np)

        final_image = Image.fromarray(np.clip(combined_np, 0, 255).astype(np.uint8))
        if original_mode != "L":
            final_image = final_image.convert(original_mode)

        if is_tensor:
            return v2.functional.to_image(final_image).to(device)
        return final_image


class CutoutAugmentation:
    """
    Zeros out a randomly placed rectangular region.
    """
    METADATA = {
        "id": "cutout",
        "name": "Cutout",
        "order": 6,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "size_ratio": {"type": "float", "min": 0.05, "max": 0.6, "default": 0.35},
        },
    }

    def __init__(self, probability: float = 0.5, size_ratio: float = 0.35):
        self.probability = probability
        self.size_ratio = size_ratio

    def __call__(self, image: Union[Image.Image, torch.Tensor], **kwargs) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image.cpu())
        else:
            img_pil = image

        w, h = img_pil.size
        ratio = kwargs.get("size_ratio", self.size_ratio)
        box_w = random.randint(int(w * 0.15), int(w * ratio))
        box_h = random.randint(int(h * 0.15), int(h * ratio))

        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)

        img_pil = img_pil.copy()
        ImageDraw.Draw(img_pil).rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=0)

        if is_tensor:
            return v2.functional.to_image(img_pil).to(device)
        return img_pil


class DataAugmenter:
    """
    Optimized Data Augmentation using torchvision.transforms.v2.
    """
    def __init__(self, config: Optional[dict] = None, is_training: bool = True):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.is_training = is_training

        self.flip = HorizontalFlipAugmentation(probability=self.config.get("flip_prob", 0.5))
        self.affine = AffineAugmentation(
            self.config.get("rotation_range", [-30, 30]),
            self.config.get("scaling_range", [0.8, 1.2]),
            self.config.get("translation", None)
        )
        self.intensity_jitter = ThermalIntensityJitter(
            probability=self.config.get("intensity_jitter_prob", 0.5),
            brightness_range=self.config.get("intensity_jitter_range", [0.55, 1.15]),
            contrast_range=self.config.get("contrast_jitter_range", [0.5, 1.15])
        )
        self.sensor_noise = IRSensorNoise(
            probability=self.config.get("sensor_noise_prob", 0.4),
            sigma_range=self.config.get("sensor_noise_sigma", [5.0, 12.0])
        )
        self.thermal_augmenter = ThermalDiffusionAugmenter(
            probability=self.config.get("occlusion_prob", 0.5),
            is_training=self.is_training
        )
        self.self_ghosting_footprint = SelfGhostingThermalFootprint(probability=self.config.get("footprint_prob", 0.5))
        self.cutout = CutoutAugmentation(
            probability=self.config.get("cutout_prob", 0.5),
            size_ratio=self.config.get("cutout_size_ratio", 0.35)
        )

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_ir: bool = False,
        return_pair: bool = False
    ) -> Any:
        if not self.enabled:
            if return_pair: return image, image, joints
            return image, joints

        # 1. Prepare Keypoints Safely
        if joints is not None:
            if torch.is_tensor(joints):
                joints_np = joints.cpu().numpy()
            else:
                joints_np = joints
                
            coords = torch.from_numpy(joints_np[:2, :].T).float().unsqueeze(0)
            
            if joints_np.shape[0] >= 3:
                vis = torch.from_numpy(joints_np[2, :]).float()
            else:
                vis = torch.ones(joints_np.shape[1], dtype=torch.float32)
                
            w, h = (image.width, image.height) if hasattr(image, "width") else (image.shape[-1], image.shape[-2])
            kpts = tv_tensors.KeyPoints(coords, canvas_size=(h, w))
        else:
            kpts = None
            vis = None

        if self.is_training:
            image, kpts = self.flip(image, kpts)

        if kpts is not None:
            image, kpts = self.affine(image, kpts)
        else:
            image, _ = self.affine(image)

        image = self.intensity_jitter(image)
        image = self.sensor_noise(image)

        source_image = image.clone() if return_pair and torch.is_tensor(image) else (image.copy() if return_pair else image)

        image = self.thermal_augmenter(image, joints=kpts, is_ir=is_ir)
        if is_ir:
            image = self.self_ghosting_footprint(image)
        image = self.cutout(image)

        if kpts is not None:
            num_kpts = kpts.shape[1]
            kpts_extracted = kpts.as_subclass(torch.Tensor) if hasattr(kpts, "as_subclass") else torch.tensor(kpts)
            final_coords = kpts_extracted.view(num_kpts, 2).T
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = joints.cpu().numpy() if torch.is_tensor(joints) else (joints.copy() if joints is not None else None)

        if return_pair:
            return image, source_image, final_joints
        return image, final_joints


def get_available_augmentations():
    augmentations = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and hasattr(obj, "METADATA"):
            augmentations.append(obj.METADATA)
    return sorted(augmentations, key=lambda x: x.get("order", 99))


def apply_custom_augmentations(image, joints, aug_list, is_ir=False):
    """
    Dynamically applies a list of augmentations with specific parameters.
    """
    all_classes = {
        obj.METADATA["id"]: obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and hasattr(obj, "METADATA")
    }

    if joints is not None:
        if torch.is_tensor(joints):
            joints_np = joints.cpu().numpy()
        else:
            joints_np = joints
            
        coords = torch.from_numpy(joints_np[:2, :].T).float().unsqueeze(0)
        if joints_np.shape[0] >= 3:
            vis = torch.from_numpy(joints_np[2, :]).float()
        else:
            vis = torch.ones(joints_np.shape[1], dtype=torch.float32)
            
        w, h = (image.width, image.height) if hasattr(image, "width") else (image.shape[-1], image.shape[-2])
        kpts = tv_tensors.KeyPoints(coords, canvas_size=(h, w))
    else:
        kpts = None

    for aug_cfg in aug_list:
        aug_id = aug_cfg.get("id")
        params = aug_cfg.get("params", {}).copy()

        if aug_id in all_classes:
            aug_cls = all_classes[aug_id]
            if "probability" not in params:
                params["probability"] = 1.0

            inst = aug_cls()

            if aug_id in ["flip", "affine"]:
                image, kpts = inst(image, kpts, **params)
            elif aug_id == "thermal_diffusion":
                image = inst(image, joints=kpts, is_ir=is_ir, **params)
            else:
                image = inst(image, **params)

    if kpts is not None:
        num_kpts = kpts.shape[1]
        kpts_extracted = kpts.as_subclass(torch.Tensor) if hasattr(kpts, "as_subclass") else torch.tensor(kpts)
        final_coords = kpts_extracted.view(num_kpts, 2).T
        if 'vis' in locals():
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = joints.copy()
            final_joints[:2, :] = final_coords.numpy()
    else:
        final_joints = joints.cpu().numpy() if torch.is_tensor(joints) else (joints.copy() if joints is not None else None)

    return image, final_joints

DataAugmenterV2 = DataAugmenter