"""
Generator architectures for CycleGAN and CUT domain translation.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, cast


class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization."""

    def __init__(self, in_features: int) -> None:
        """
        Initializes the ResidualBlock.

        Args:
            in_features: Number of input and output channels.
        """
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + cast(torch.Tensor, self.block(x))


class GeneratorResNet(nn.Module):
    """
    ResNet-based Generator.
    Uses Reflection Padding to reduce boundary artifacts.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_residual_blocks: int = 9,
        pretrained: bool = False,
    ) -> None:
        """
        Initializes the generator.

        Args:
            input_shape: Shape of the input image (C, H, W).
            num_residual_blocks: Number of residual blocks in the bottleneck.
            pretrained: If True, attempts to load ImageNet weights into the encoder.
        """
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        initial: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        downsampling: List[nn.Module] = []
        for _ in range(2):
            out_features *= 2
            downsampling += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        self.encoder = nn.Sequential(*(initial + downsampling))

        # Residual blocks
        resblocks: List[nn.Module] = []
        for _ in range(num_residual_blocks):
            resblocks += [ResidualBlock(out_features)]
        self.resblocks = nn.Sequential(*resblocks)

        # Upsampling
        upsampling: List[nn.Module] = []
        for _ in range(2):
            out_features //= 2
            upsampling += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer (Forces monochromatic output to prevent color hallucinations)
        upsampling += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, 1, 7),
            nn.Tanh(),
        ]

        self.decoder = nn.Sequential(*upsampling)

        if pretrained:
            self._load_pretrained_encoder()

    def _load_pretrained_encoder(self) -> None:
        """Loads ImageNet weights into the encoder layers."""
        try:
            import torchvision.models as models

            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("[Generator] Initializing encoder with ResNet18 ImageNet weights...")
            with torch.no_grad():
                # Map 7x7 conv (first layer in both)
                first_conv = self.encoder[1]
                if isinstance(first_conv, nn.Conv2d):
                    first_conv.weight.copy_(resnet.conv1.weight)
        except Exception as e:
            print(f"[Generator] Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor.

        Returns:
            Translated image tensor.
        """
        x = cast(torch.Tensor, self.encoder(x))
        x = cast(torch.Tensor, self.resblocks(x))
        x = cast(torch.Tensor, self.decoder(x))
        # Replicate to 3 channels to maintain compatibility with HRNet/ViTPose
        # and prevent "color hallucinations" (R!=G!=B) in thermal domain.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x
