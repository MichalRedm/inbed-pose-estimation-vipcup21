"""
Custom neural network layers for pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SoftArgmax2D(nn.Module):
    """
    Differentiable Argmax for 2D heatmaps.
    Converts (B, J, H, W) heatmaps to (B, J, 2) coordinates.
    """

    grid_x: torch.Tensor
    grid_y: torch.Tensor

    def __init__(
        self,
        base_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        temperature: float = 10.0,
    ) -> None:
        """
        Initializes SoftArgmax2D.

        Args:
            base_size: The target image resolution (Height, Width).
            heatmap_size: The input heatmap resolution (Height, Width).
            temperature: Temperature parameter for softmax scaling.
        """
        super().__init__()
        self.base_size = base_size
        self.heatmap_size = heatmap_size
        self.temperature = temperature

        # Create coordinate grids
        grid_x = torch.arange(heatmap_size[0]).float()
        grid_y = torch.arange(heatmap_size[1]).float()

        # Scale to base size (256x256)
        grid_x = grid_x * (base_size[0] / heatmap_size[0])
        grid_y = grid_y * (base_size[1] / heatmap_size[1])

        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Performs differentiable coordinate decoding.

        Args:
            heatmaps: Input heatmaps of shape (B, J, H, W).

        Returns:
            Coordinates of shape (B, J, 2) where 2 is (x, y).
        """
        B, J, H, W = heatmaps.shape

        # Apply temperature-scaled softmax to each heatmap to get probability distributions
        # Flatten H, W first: (B, J, H*W)
        flat_heatmaps = heatmaps.view(B, J, -1)
        probs = F.softmax(flat_heatmaps * self.temperature, dim=-1)
        probs = probs.view(B, J, H, W)

        # Compute expected x and y
        # Expected X: sum over Y, then dot product with grid_x
        # Expected Y: sum over X, then dot product with grid_y
        marginal_x = torch.sum(probs, dim=2)  # (B, J, W)
        marginal_y = torch.sum(probs, dim=3)  # (B, J, H)

        expected_x = torch.sum(marginal_x * self.grid_x, dim=2)  # (B, J)
        expected_y = torch.sum(marginal_y * self.grid_y, dim=2)  # (B, J)

        return torch.stack([expected_x, expected_y], dim=-1)
