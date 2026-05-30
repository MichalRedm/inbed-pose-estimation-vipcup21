import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
from typing import Union, Dict, Any, List, cast


class ThermalIntensityJitter:
    """
    Randomly dampens or slightly boosts contrast and brightness to simulate thermal attenuation.
    """

    METADATA: Dict[str, Any] = {
        "id": "intensity_jitter",
        "name": "Thermal Intensity Jitter",
        "order": 2,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "brightness": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0},
            "contrast": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0},
        },
    }

    probability: float
    brightness_range: List[float]
    contrast_range: List[float]

    def __init__(
        self,
        probability: float = 0.5,
        brightness_range: List[float] = [0.55, 1.15],
        contrast_range: List[float] = [0.5, 1.15],
    ) -> None:
        self.probability = probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self, image: Union[Image.Image, torch.Tensor], **kwargs: Any
    ) -> Union[Image.Image, torch.Tensor]:
        prob = float(kwargs.get("probability", self.probability))
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            img_tensor = cast(torch.Tensor, image)
            device = img_tensor.device
            img_pil = v2.functional.to_pil_image(img_tensor)
        else:
            img_pil = cast(Image.Image, image)

        img_np = np.array(img_pil).astype(np.float32)

        if "brightness" in kwargs:
            scale_b = float(kwargs["brightness"])
        else:
            scale_b = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img_np = img_np * scale_b

        if "contrast" in kwargs:
            scale_c = float(kwargs["contrast"])
        else:
            scale_c = random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = img_np.mean()
        img_np = (img_np - mean) * scale_c + mean

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        res_pil = Image.fromarray(img_np)

        if is_tensor:
            return cast(torch.Tensor, v2.functional.to_image(res_pil).to(device))
        return res_pil


class IRSensorNoise:
    """
    Simulates readout thermal/Gaussian noise and dead/hot pixels.
    """

    METADATA: Dict[str, Any] = {
        "id": "sensor_noise",
        "name": "IR Sensor Noise",
        "order": 3,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "sigma": {"type": "float", "min": 0.0, "max": 30.0, "default": 8.0},
            "sp_prob": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.003},
        },
    }

    probability: float
    sigma_range: List[float]
    sp_prob: float

    def __init__(
        self,
        probability: float = 0.4,
        sigma_range: List[float] = [5.0, 12.0],
        sp_prob: float = 0.003,
    ) -> None:
        self.probability = probability
        self.sigma_range = sigma_range
        self.sp_prob = sp_prob

    def __call__(
        self, image: Union[Image.Image, torch.Tensor], **kwargs: Any
    ) -> Union[Image.Image, torch.Tensor]:
        prob = float(kwargs.get("probability", self.probability))
        if random.random() > prob:
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            img_tensor = cast(torch.Tensor, image)
            device = img_tensor.device
            img_pil = v2.functional.to_pil_image(img_tensor)
        else:
            img_pil = cast(Image.Image, image)

        img_np = np.array(img_pil).astype(np.float32)

        if "sigma" in kwargs:
            sigma = float(kwargs["sigma"])
        else:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        img_np = img_np + noise

        current_sp_prob = float(kwargs.get("sp_prob", self.sp_prob))
        sp_mask = np.random.random(img_np.shape[:2])
        img_np[sp_mask < (current_sp_prob / 2.0)] = 255.0
        img_np[sp_mask > (1.0 - current_sp_prob / 2.0)] = 0.0

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        res_pil = Image.fromarray(img_np)

        if is_tensor:
            return cast(torch.Tensor, v2.functional.to_image(res_pil).to(device))
        return res_pil
