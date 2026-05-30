import random
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.v2 as v2
from typing import Union, Dict, Any, cast


class CutoutAugmentation:
    """
    Zeros out a randomly placed rectangular region.
    """

    METADATA: Dict[str, Any] = {
        "id": "cutout",
        "name": "Cutout",
        "order": 5,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "size_ratio": {"type": "float", "min": 0.05, "max": 0.6, "default": 0.35},
        },
    }

    probability: float
    size_ratio: float

    def __init__(self, probability: float = 0.5, size_ratio: float = 0.35) -> None:
        self.probability = probability
        self.size_ratio = size_ratio

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

        w, h = img_pil.size
        ratio = float(kwargs.get("size_ratio", self.size_ratio))
        box_w = random.randint(int(w * 0.15), int(w * ratio))
        box_h = random.randint(int(h * 0.15), int(h * ratio))

        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)

        res_pil = img_pil.copy()
        draw = ImageDraw.Draw(res_pil)
        draw.rectangle(
            (float(x1), float(y1), float(x1 + box_w), float(y1 + box_h)), fill=0
        )

        if is_tensor:
            return v2.functional.to_image(res_pil).to(device)  # type: ignore[no-any-return]
        return res_pil
