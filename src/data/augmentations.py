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
        self, image: Union[Image.Image, torch.Tensor], **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        if "brightness" in kwargs:
            scale_b = kwargs["brightness"]
        else:
            scale_b = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img_np = img_np * scale_b

        if "contrast" in kwargs:
            scale_c = kwargs["contrast"]
        else:
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
        self, image: Union[Image.Image, torch.Tensor], **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        img_np = np.array(img_pil).astype(np.float32)

        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
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

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[torch.Tensor] = None,
        is_ir: bool = True,
        **kwargs,
    ) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if not is_ir or random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        w, h = img_pil.size

        # 1. Determine blanket Y start position (base_y)
        # Head (index 13) and neck/thorax (index 12) should ALWAYS remain uncovered!
        head_y = None
        shoulders_y = []
        hips_y = []
        knees_y = []

        if joints is not None:
            # Handle keypoints tensor format
            if torch.is_tensor(joints):
                if len(joints.shape) == 3:  # (1, N, 2)
                    j_np = joints[0].cpu().numpy()
                elif len(joints.shape) == 2:  # (N, 2) or (3, N)
                    if joints.shape[0] == 3:
                        j_np = joints[:2, :].T.cpu().numpy()
                    else:
                        j_np = joints.cpu().numpy()
                else:
                    j_np = np.array(joints)
            else:
                j_np = np.array(joints)

            if len(j_np.shape) == 2 and j_np.shape[0] >= 14:
                if j_np[13, 0] > 0 or j_np[13, 1] > 0:
                    head_y = j_np[13, 1]
                
                for s_idx in [8, 9]:
                    if j_np[s_idx, 0] > 0 or j_np[s_idx, 1] > 0:
                        shoulders_y.append(j_np[s_idx, 1])
                
                for h_idx in [2, 3]:
                    if j_np[h_idx, 0] > 0 or j_np[h_idx, 1] > 0:
                        hips_y.append(j_np[h_idx, 1])

                for k_idx in [1, 4]:
                    if j_np[k_idx, 0] > 0 or j_np[k_idx, 1] > 0:
                        knees_y.append(j_np[k_idx, 1])

        # Enforce minimum Y to protect the head
        min_allowed_y = int(h * 0.15)
        if head_y is not None:
            min_allowed_y = max(min_allowed_y, int(head_y + 15))

        if "base_y_ratio" in kwargs:
            base_y = int(kwargs["base_y_ratio"] * h)
            base_y = max(min_allowed_y, base_y)
        else:
            # We want to randomly place the blanket covering either:
            # 1. Chest down (shoulders area)
            # 2. Waist down (hips area)
            # 3. Thighs/knees down (knees area)
            coverage_choices = []
            
            if shoulders_y:
                coverage_choices.append(min(shoulders_y))
            if hips_y:
                coverage_choices.append(min(hips_y))
            if knees_y:
                coverage_choices.append(min(knees_y))

            if coverage_choices:
                base_y = random.choice(coverage_choices)
                base_y += random.randint(-15, 15)
            else:
                base_y = random.randint(int(h * 0.25), int(h * 0.7))

            base_y = max(min_allowed_y, min(base_y, h - 20))

        base_y = int(base_y)

        # 2. Create wavy mask to simulate natural blanket edge
        mask = Image.new("L", img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        wavy_points = []
        num_points = 20
        freq = random.uniform(1.5, 3.5)
        amp = random.uniform(4, 15)
        phase = random.uniform(0, 2 * np.pi)

        for i in range(num_points + 1):
            x = int(i * w / num_points)
            y = int(base_y + amp * np.sin(freq * (x / w) * 2 * np.pi + phase))
            y = max(0, min(h - 1, y))
            wavy_points.append((x, y))

        polygon_points = list(wavy_points)
        polygon_points.append((w, h))
        polygon_points.append((0, h))
        draw.polygon(polygon_points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 7)))

        # 3. Get image as numpy and estimate ambient background temperature of the bed
        img_np = np.array(img_pil).astype(np.float32)
        ambient_est = np.percentile(img_np, 15)

        # 4. Generate large-scale blanket drape mask (simulate large folds and valleys)
        drape_mask = Image.new("L", img_pil.size, 255)
        drape_draw = ImageDraw.Draw(drape_mask)
        num_large_folds = random.randint(3, 6)
        for _ in range(num_large_folds):
            fx1 = random.randint(-40, w + 40)
            fy1 = random.randint(base_y - 40, h + 40)
            fx2 = random.randint(-40, w + 40)
            fy2 = fy1 + random.randint(80, 250)
            cx = random.randint(min(fx1, fx2) - 50, max(fx1, fx2) + 50)
            cy = (fy1 + fy2) // 2 + random.randint(-20, 20)
            
            pts = []
            for t in np.linspace(0, 1, 15):
                px = int((1-t)**2 * fx1 + 2*(1-t)*t * cx + t**2 * fx2)
                py = int((1-t)**2 * fy1 + 2*(1-t)*t * cy + t**2 * fy2)
                pts.append((px, py))
            
            fold_val = random.randint(40, 120)
            fold_width = random.randint(25, 50)
            drape_draw.line(pts, fill=fold_val, width=fold_width, joint="round")
        
        drape_blur = drape_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(10.0, 18.0)))
        drape_np = np.array(drape_blur).astype(np.float32) / 255.0

        # 4a. Generate small-scale wrinkles mask (simulate sharp fine folds)
        wrinkle_mask = Image.new("L", img_pil.size, 255)
        wrinkle_draw = ImageDraw.Draw(wrinkle_mask)
        num_small_folds = random.randint(4, 8)
        for _ in range(num_small_folds):
            fx1 = random.randint(-40, w + 40)
            fy1 = random.randint(base_y - 20, h + 40)
            fx2 = fx1 + random.randint(-60, 60)
            fy2 = fy1 + random.randint(40, 180)
            cx = (fx1 + fx2) // 2 + random.randint(-15, 15)
            cy = (fy1 + fy2) // 2 + random.randint(-10, 10)
            
            pts = []
            for t in np.linspace(0, 1, 10):
                px = int((1-t)**2 * fx1 + 2*(1-t)*t * cx + t**2 * fx2)
                py = int((1-t)**2 * fy1 + 2*(1-t)*t * cy + t**2 * fy2)
                pts.append((px, py))
            
            fold_val = random.randint(80, 160)
            fold_width = random.randint(3, 7)
            wrinkle_draw.line(pts, fill=fold_val, width=fold_width, joint="round")
            
        wrinkle_blur = wrinkle_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.5)))
        wrinkle_np = np.array(wrinkle_blur).astype(np.float32) / 255.0

        # Combine large-scale drapes and small-scale wrinkles
        combined_drape_np = drape_np * wrinkle_np

        # 5. Create blanket base fabric layer (dampened ambient with wrinkles and noise)
        blanket_ambient = ambient_est + (combined_drape_np - 0.85) * 12.0
        noise = np.random.normal(0, 1.5, img_np.shape).astype(np.float32)
        blanket_base = blanket_ambient + noise

        # 6. Simulate physical body heat transmission through blanket
        body_heat = np.maximum(img_np - ambient_est, 0.0)
        
        # Apply non-linear boost to emphasize hottest contact points
        body_heat_norm = body_heat / (np.max(body_heat) + 1e-5)
        body_heat_boosted = np.power(body_heat_norm, 1.3) * np.max(body_heat)
        heat_pil = Image.fromarray(np.clip(body_heat_boosted, 0, 255).astype(np.uint8))

        # Heat bloom and contact diffusion (scaled down to avoid the "too blurry" issue)
        bloom_radius = random.uniform(8.0, 16.0)
        contact_radius = random.uniform(2.5, 5.0)
        
        bloom_img = heat_pil.filter(ImageFilter.GaussianBlur(radius=bloom_radius))
        contact_img = heat_pil.filter(ImageFilter.GaussianBlur(radius=contact_radius))
        
        bloom_np = np.array(bloom_img).astype(np.float32)
        contact_np = np.array(contact_img).astype(np.float32)

        # Mix bloom and contact based on the combined drape topology
        mixed_heat_np = bloom_np * (1.0 - combined_drape_np) + contact_np * combined_drape_np

        # 7. Apply dampening factors
        if "damp_factor" in kwargs:
            damp_factor = kwargs["damp_factor"]
        else:
            # We want slightly higher heat transmission so the body under the cover remains visible
            damp_factor = random.uniform(0.18, 0.42)

        # Valleys attenuate the heat more heavily
        transmitted_heat = mixed_heat_np * damp_factor * combined_drape_np
        dampened_np = blanket_base + transmitted_heat

        # 8. Draw a soft drop-shadow crease along the blanket edge (top-line only!)
        shadow_mask = Image.new("L", img_pil.size, 0)
        shadow_draw = ImageDraw.Draw(shadow_mask)
        shadow_draw.line(wavy_points, fill=255, width=random.randint(4, 8), joint="round")
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))
        shadow_np = np.array(shadow_mask).astype(np.float32) / 255.0
        dampened_np = dampened_np * (1.0 - shadow_np * random.uniform(0.03, 0.08))

        # Convert back to PIL Image
        dampened = Image.fromarray(np.clip(dampened_np, 0, 255).astype(np.uint8))
        
        # Blanket layer: composite the blanket area with the original uncovered image using the wavy mask
        final_image = Image.composite(dampened, img_pil, mask)

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
        "order": 5,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "size_ratio": {"type": "float", "min": 0.05, "max": 0.6, "default": 0.35},
        },
    }

    def __init__(self, probability: float = 0.5, size_ratio: float = 0.35):
        self.probability = probability
        self.size_ratio = size_ratio

    def __call__(
        self, image: Union[Image.Image, torch.Tensor], **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        prob = kwargs.get("probability", self.probability)
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        w, h = img_pil.size
        ratio = kwargs.get("size_ratio", self.size_ratio)
        box_w = random.randint(int(w * 0.15), int(w * ratio))
        box_h = random.randint(int(h * 0.15), int(h * ratio))

        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)

        img_pil = img_pil.copy()
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=0)

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

        # Components
        self.flip = HorizontalFlipAugmentation(
            probability=self.config.get("flip_prob", 0.5)
        )

        rot_range = self.config.get("rotation_range", [-30, 30])
        scale_range = self.config.get("scaling_range", [0.8, 1.2])
        translate = self.config.get("translation", None)
        self.affine = AffineAugmentation(rot_range, scale_range, translate)

        self.intensity_jitter = ThermalIntensityJitter(
            probability=self.config.get("intensity_jitter_prob", 0.5),
            brightness_range=self.config.get("intensity_jitter_range", [0.55, 1.15]),
            contrast_range=self.config.get("contrast_jitter_range", [0.5, 1.15]),
        )

        self.sensor_noise = IRSensorNoise(
            probability=self.config.get("sensor_noise_prob", 0.4),
            sigma_range=self.config.get("sensor_noise_sigma", [5.0, 12.0]),
        )

        self.thermal_augmenter = ThermalDiffusionAugmenter(
            probability=self.config.get("occlusion_prob", 0.5),
            is_training=self.is_training,
        )

        self.cutout = CutoutAugmentation(
            probability=self.config.get("cutout_prob", 0.5),
            size_ratio=self.config.get("cutout_size_ratio", 0.35),
        )

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

        # 2. Sequential Application
        if self.is_training:
            image, kpts = self.flip(image, kpts)

        if kpts is not None:
            image, kpts = self.affine(image, kpts)
        else:
            image, _ = self.affine(image)

        image = self.intensity_jitter(image)
        image = self.sensor_noise(image)

        source_image = image
        if return_pair:
            source_image = image.clone() if torch.is_tensor(image) else image.copy()

        image = self.thermal_augmenter(image, joints=kpts, is_ir=is_ir)
        image = self.cutout(image)

        # 3. Final Assembly
        if kpts is not None:
            num_kpts = kpts.shape[1]
            final_coords = kpts.view(num_kpts, 2).T  # (2, num_kpts)
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = None

        if return_pair:
            return image, source_image, final_joints
        return image, final_joints


def get_available_augmentations():
    """
    Returns metadata for all discoverable augmentation classes in this module.
    """
    augmentations = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and hasattr(obj, "METADATA"):
            augmentations.append(obj.METADATA)

    return sorted(augmentations, key=lambda x: x.get("order", 99))


def apply_custom_augmentations(image, joints, aug_list, is_ir=False):
    """
    Dynamically applies a list of augmentations with specific parameters.
    aug_list: List of {"id": "...", "params": {...}}
    """
    # Initialize classes once if needed, or just create instances on the fly
    # To be fast, we find the classes
    all_classes = {
        obj.METADATA["id"]: obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and hasattr(obj, "METADATA")
    }

    # Prepare kpts if needed
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

    for aug_cfg in aug_list:
        aug_id = aug_cfg.get("id")
        params = aug_cfg.get("params", {})

        if aug_id in all_classes:
            aug_cls = all_classes[aug_id]
            # Create a temporary instance with default config (params will override in __call__)
            # Note: For visualizer, we usually want probability=1.0 if not specified
            if "probability" not in params:
                params["probability"] = 1.0

            inst = aug_cls()

            if aug_id in ["flip", "affine"]:
                image, kpts = inst(image, kpts, **params)
            elif aug_id == "thermal_diffusion":
                image = inst(image, joints=kpts, is_ir=is_ir, **params)
            else:
                image = inst(image, **params)

    # Final assembly
    if kpts is not None:
        num_kpts = kpts.shape[1]
        final_coords = kpts.view(num_kpts, 2).T  # (2, num_kpts)
        if "vis" in locals():
            # If vis was updated (e.g. by flip)
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            # Fallback
            final_joints = joints.copy()
            final_joints[:2, :] = final_coords.numpy()
    else:
        final_joints = None

    return image, final_joints


# Legacy alias
DataAugmenterV2 = DataAugmenter
