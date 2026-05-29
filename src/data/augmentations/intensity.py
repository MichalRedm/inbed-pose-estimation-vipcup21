import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
from typing import Union


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
