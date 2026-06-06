"""
Domain translation data augmentations using GAN-based style transfer.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from PIL import Image
from typing import Union, Dict, Any, Optional, cast, Tuple


class CycleGANAugmentation:
    """
    CycleGAN-based style transfer for data augmentation.
    Translates 'uncovered' IR images to 'covered' IR domains using a pre-trained Generator.
    """

    METADATA: Dict[str, Any] = {
        "id": "cyclegan",
        "name": "CycleGAN Target Translation",
        "description": "Translates clean IR subjects to look like they are under a blanket using a trained CycleGAN generator.",
        "order": 40,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}
        },
    }

    probability: float
    enabled: bool
    device: torch.device
    generator: Optional[nn.Module]
    checkpoint_path: str
    alpha_blend: bool
    alpha_range: Tuple[float, float]

    def __init__(
        self,
        probability: float = 0.5,
        checkpoint_path: str = "models/cyclegan_gen_A2B.pth",
        alpha_blend: bool = False,
        alpha_range: Tuple[float, float] = (0.6, 1.0),
    ) -> None:
        """
        Initializes the CycleGAN augmentation.

        Args:
            probability: Probability of applying the translation.
            checkpoint_path: Path to the generator checkpoint.
            alpha_blend: Whether to blend the translated image with the original.
            alpha_range: Range (min, max) for blending weights.
        """
        self.probability = probability
        self.enabled = probability > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.checkpoint_path = checkpoint_path
        self.alpha_blend = alpha_blend
        self.alpha_range = alpha_range

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
                                checkpoint_path,
                                map_location=self.device,
                                weights_only=True,
                            ),
                            strict=False,  # Allow mismatched keys to prevent crash if architecture changed
                        )
                        self.generator = self.generator.to(self.device)
                        self.generator.eval()
                        print(f"CycleGAN generator loaded from {checkpoint_path}")
                    except Exception as e:
                        print(
                            f"Warning: Failed to load CycleGAN checkpoint weights: {e}"
                        )
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
        self, img: Union[Image.Image, torch.Tensor], **kwargs: Any
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Applies CycleGAN translation.

        Args:
            img: Input image.
            **kwargs: Override for probability and force_apply.

        Returns:
            Translated image.
        """
        force = bool(kwargs.get("force_apply", False))
        prob = float(kwargs.get("probability", self.probability))
        if (
            not self.enabled
            or self.generator is None
            or (not force and torch.rand(1).item() > prob)
        ):
            return img

        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_tensor = cast(torch.Tensor, img)
            img_t = (
                img_tensor
                if img_tensor.dtype == torch.float32
                else img_tensor.float() / 255.0
            )
        else:
            img_pil = cast(Image.Image, img)
            img_t = cast(torch.Tensor, v2.functional.to_image(img_pil)).float() / 255.0

        original_channels = img_t.shape[0]

        # Keep original [0, 1] tensor on target device for blending
        img_t_orig = img_t.to(self.device)

        # Generator expects batch dimension and normalized [-1, 1]
        img_t_norm = (img_t_orig * 2) - 1.0
        img_t_norm = img_t_norm.unsqueeze(0)

        # Ensure 3 channels for the generator
        if img_t_norm.shape[1] == 1:
            img_t_norm = img_t_norm.repeat(1, 3, 1, 1)

        with torch.no_grad():
            fake_target = self.generator(img_t_norm)

        # Denormalize [0, 1]
        fake_target = (fake_target.squeeze(0) + 1.0) / 2.0

        # Convert back to original channel count if needed
        if original_channels == 1 and fake_target.shape[0] == 3:
            # We can average or take the first channel. Taking first is common.
            fake_target = fake_target[0:1, :, :]

        fake_target = torch.clamp(fake_target, 0, 1)

        if self.alpha_blend:
            alpha = float(
                torch.empty(1).uniform_(self.alpha_range[0], self.alpha_range[1]).item()
            )
            fake_target = alpha * fake_target + (1.0 - alpha) * img_t_orig

        if not is_tensor:
            return v2.functional.to_pil_image(fake_target.cpu())  # type: ignore[no-any-return]

        # Match input tensor format (if it was [0, 255] unscaled)
        img_input = cast(torch.Tensor, img)
        if img_input.max() > 1.0:
            fake_target = fake_target * 255.0
        return fake_target.to(img_input.device)


class CUTAugmentation:
    """
    CUT-based style transfer for data augmentation.
    Translates 'uncovered' IR images to 'covered' IR domains using a pre-trained Generator.
    """

    METADATA: Dict[str, Any] = {
        "id": "cut",
        "name": "CUT Target Translation",
        "description": "Translates clean IR subjects to look like they are under a blanket using a trained CUT generator.",
        "order": 41,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}
        },
    }

    probability: float
    enabled: bool
    device: torch.device
    generator: Optional[nn.Module]
    checkpoint_path: str
    alpha_blend: bool
    alpha_range: Tuple[float, float]

    def __init__(
        self,
        probability: float = 0.5,
        checkpoint_path: str = "models/cut_gen.pth",
        alpha_blend: bool = False,
        alpha_range: Tuple[float, float] = (0.6, 1.0),
    ) -> None:
        """
        Initializes the CUT augmentation.

        Args:
            probability: Probability of applying the translation.
            checkpoint_path: Path to the generator checkpoint.
            alpha_blend: Whether to blend the translated image with the original.
            alpha_range: Range (min, max) for blending weights.
        """
        self.probability = probability
        self.enabled = probability > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.checkpoint_path = checkpoint_path
        self.alpha_blend = alpha_blend
        self.alpha_range = alpha_range

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
                                checkpoint_path,
                                map_location=self.device,
                                weights_only=True,
                            ),
                            strict=False,  # Allow mismatched keys to prevent crash if architecture changed
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
        self, img: Union[Image.Image, torch.Tensor], **kwargs: Any
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Applies CUT translation.

        Args:
            img: Input image.
            **kwargs: Override for probability and force_apply.

        Returns:
            Translated image.
        """
        force = bool(kwargs.get("force_apply", False))
        prob = float(kwargs.get("probability", self.probability))
        if (
            not self.enabled
            or self.generator is None
            or (not force and torch.rand(1).item() > prob)
        ):
            return img

        is_tensor = torch.is_tensor(img)
        if is_tensor:
            img_tensor = cast(torch.Tensor, img)
            img_t = (
                img_tensor
                if img_tensor.dtype == torch.float32
                else img_tensor.float() / 255.0
            )
        else:
            img_pil = cast(Image.Image, img)
            img_t = cast(torch.Tensor, v2.functional.to_image(img_pil)).float() / 255.0

        original_channels = img_t.shape[0]

        # Keep original [0, 1] tensor on target device for blending
        img_t_orig = img_t.to(self.device)

        # Generator expects batch dimension and normalized [-1, 1]
        img_t_norm = (img_t_orig * 2) - 1.0
        img_t_norm = img_t_norm.unsqueeze(0)

        # Ensure 3 channels for the generator
        if img_t_norm.shape[1] == 1:
            img_t_norm = img_t_norm.repeat(1, 3, 1, 1)

        with torch.no_grad():
            fake_target = self.generator(img_t_norm)

        # Denormalize [0, 1]
        fake_target = (fake_target.squeeze(0) + 1.0) / 2.0

        # Convert back to original channel count if needed
        if original_channels == 1 and fake_target.shape[0] == 3:
            # We can average or take the first channel. Taking first is common.
            fake_target = fake_target[0:1, :, :]

        fake_target = torch.clamp(fake_target, 0, 1)

        if self.alpha_blend:
            alpha = float(
                torch.empty(1).uniform_(self.alpha_range[0], self.alpha_range[1]).item()
            )
            fake_target = alpha * fake_target + (1.0 - alpha) * img_t_orig

        if not is_tensor:
            return v2.functional.to_pil_image(fake_target.cpu())  # type: ignore[no-any-return]

        # Match input tensor format (if it was [0, 255] unscaled)
        img_input = cast(torch.Tensor, img)
        if img_input.max() > 1.0:
            fake_target = fake_target * 255.0
        return fake_target.to(img_input.device)
