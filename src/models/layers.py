import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):
    """
    Differentiable Argmax for 2D heatmaps.
    Converts (B, J, H, W) heatmaps to (B, J, 2) coordinates.
    """

    def __init__(self, base_size=(256, 256), heatmap_size=(64, 64)):
        super().__init__()
        self.base_size = base_size
        self.heatmap_size = heatmap_size

        # Create coordinate grids
        # (W,) and (H,)
        grid_x = torch.arange(heatmap_size[0]).float()
        grid_y = torch.arange(heatmap_size[1]).float()

        # Scale to base size (256x256)
        grid_x = grid_x * (base_size[0] / heatmap_size[0])
        grid_y = grid_y * (base_size[1] / heatmap_size[1])

        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def forward(self, heatmaps):
        """
        heatmaps: (B, J, H, W)
        Returns: (B, J, 2) where 2 is (x, y)
        """
        B, J, H, W = heatmaps.shape

        # Apply softmax to each heatmap to get probability distributions
        # Flatten H, W first: (B, J, H*W)
        flat_heatmaps = heatmaps.view(B, J, -1)
        probs = F.softmax(flat_heatmaps, dim=-1)
        probs = probs.view(B, J, H, W)

        # Compute expected x and y
        # Expected X: sum over Y, then dot product with grid_x
        # Expected Y: sum over X, then dot product with grid_y
        marginal_x = torch.sum(probs, dim=2)  # (B, J, W)
        marginal_y = torch.sum(probs, dim=3)  # (B, J, H)

        expected_x = torch.sum(marginal_x * self.grid_x, dim=2)  # (B, J)
        expected_y = torch.sum(marginal_y * self.grid_y, dim=2)  # (B, J)

        return torch.stack([expected_x, expected_y], dim=-1)
