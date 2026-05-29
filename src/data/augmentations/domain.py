import os
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from typing import Union

class CycleGANAugmentation:
    """
    CycleGAN-based style transfer for data augmentation.
    Translates 'uncovered' IR images to 'covered' IR domains using a pre-trained Generator.
    """

    METADATA = {
        "id": "cyclegan",
        "name": "CycleGAN Target Translation",
        "description": "Translates clean IR subjects to look like they are under a blanket using a trained CycleGAN generator.",
        "order": 40,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}
        },
    }

    def __init__(
        self,
        probability: float = 0.5,
        checkpoint_path: str = "models/cyclegan_gen_A2B.pth",
    ):
        self.probability = probability
        self.enabled = probability > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.checkpoint_path = checkpoint_path

        if self.enabled:
            # Try to load cyclegan generator late to avoid cyclic imports
            try:
                from src.models.cyclegan.generator import GeneratorResNet

                self.generator = GeneratorResNet(
                    (3, 256, 256)
                )  # Assuming RGB/Replicated channels
                if os.path.exists(checkpoint_path):
                    try:
                        self.generator.load_state_dict(
                            torch.load(
                                checkpoint_path, map_location=self.device, weights_only=True
                            ),
                            strict=False # Allow mismatched keys to prevent crash if architecture changed
                        )
                        self.generator = self.generator.to(self.device)
                        self.generator.eval()
                        print(f"CycleGAN generator loaded from {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load CycleGAN checkpoint weights: {e}")
                        self.enabled = False
                else:
                    print(
                        f"Warning: CycleGAN checkpoint {checkpoint_path} not found. Augmentation will be a no-op."
                    )
                    self.enabled = False
            except ImportError:
                print("Failed to import GeneratorResNet.")
                self.enabled = False

    def __call__(
        self, img: Union[Image.Image, torch.Tensor], **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        force = kwargs.get("force_apply", False)
        prob = kwargs.get("probability", self.probability)
        if not self.enabled or (not force and torch.rand(1).item() > prob):
            return img

        is_tensor = torch.is_tensor(img)
        img_t = img if is_tensor else v2.functional.to_image(img).float() / 255.0

        # Generator expects batch dimension and normalized [-1, 1]
        img_t = (img_t * 2) - 1.0
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_target = self.generator(img_t)

        # Denormalize [0, 1]
        fake_target = (fake_target.squeeze(0) + 1.0) / 2.0
        fake_target = torch.clamp(fake_target, 0, 1)

        if not is_tensor:
            return v2.functional.to_pil_image(fake_target.cpu())

        # Match input tensor format (if it was [0, 255] unscaled)
        if img.max() > 1.0:
            fake_target = fake_target * 255.0
        return fake_target.to(img.device)


class CUTAugmentation:
    """
    CUT-based style transfer for data augmentation.
    Translates 'uncovered' IR images to 'covered' IR domains using a pre-trained Generator.
    """

    METADATA = {
        "id": "cut",
        "name": "CUT Target Translation",
        "description": "Translates clean IR subjects to look like they are under a blanket using a trained CUT generator.",
        "order": 41,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}
        },
    }

    def __init__(
        self,
        probability: float = 0.5,
        checkpoint_path: str = "models/cut_gen.pth",
    ):
        self.probability = probability
        self.enabled = probability > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.checkpoint_path = checkpoint_path

        if self.enabled:
            try:
                from src.models.cyclegan.generator import GeneratorResNet

                self.generator = GeneratorResNet(
                    (3, 256, 256)
                )  # Assuming RGB/Replicated channels
                if os.path.exists(checkpoint_path):
                    try:
                        self.generator.load_state_dict(
                            torch.load(
                                checkpoint_path, map_location=self.device, weights_only=True
                            ),
                            strict=False # Allow mismatched keys to prevent crash if architecture changed
                        )
                        self.generator = self.generator.to(self.device)
                        self.generator.eval()
                        print(f"CUT generator loaded from {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load CUT checkpoint weights: {e}")
                        self.enabled = False
                else:
                    print(
                        f"Warning: CUT checkpoint {checkpoint_path} not found. Augmentation will be a no-op."
                    )
                    self.enabled = False
            except ImportError:
                print("Failed to import GeneratorResNet.")
                self.enabled = False

    def __call__(
        self, img: Union[Image.Image, torch.Tensor], **kwargs
    ) -> Union[Image.Image, torch.Tensor]:
        force = kwargs.get("force_apply", False)
        prob = kwargs.get("probability", self.probability)
        if not self.enabled or (not force and torch.rand(1).item() > prob):
            return img

        is_tensor = torch.is_tensor(img)
        img_t = img if is_tensor else v2.functional.to_image(img).float() / 255.0

        # Generator expects batch dimension and normalized [-1, 1]
        img_t = (img_t * 2) - 1.0
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_target = self.generator(img_t)

        # Denormalize [0, 1]
        fake_target = (fake_target.squeeze(0) + 1.0) / 2.0
        fake_target = torch.clamp(fake_target, 0, 1)

        if not is_tensor:
            return v2.functional.to_pil_image(fake_target.cpu())

        # Match input tensor format (if it was [0, 255] unscaled)
        if img.max() > 1.0:
            fake_target = fake_target * 255.0
        return fake_target.to(img.device)
