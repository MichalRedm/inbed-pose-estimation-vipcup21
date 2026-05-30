import torch
import torch.nn as nn
from typing import Tuple, List, cast


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies if 70x70 patches are real or fake.
    """

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(
            in_filters: int, out_filters: int, normalize: bool = True
        ) -> List[nn.Module]:
            """Returns downsampling layers of each discriminator block"""
            layers: List[nn.Module] = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(img))
