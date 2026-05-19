import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):
    """
    Differentiable Argmax for 2D heatmaps.
    Converts (B, J, H, W) heatmaps to (B, J, 2) coordinates.
    Supports optional localized window masking to prevent global center-of-mass pull (joint coalescence).
    """

    def __init__(self, base_size=(256, 256), heatmap_size=(64, 64), temperature=10.0, window_size=None):
        super().__init__()
        self.base_size = base_size
        self.heatmap_size = heatmap_size
        self.temperature = temperature
        self.window_size = window_size

        # Create coordinate grids
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
        device = heatmaps.device

        # If window_size is specified, perform local window masking around peak
        if self.window_size is not None:
            flat_heatmaps = heatmaps.view(B, J, -1)
            max_idx = flat_heatmaps.argmax(dim=-1) # (B, J)
            
            y_peak = max_idx // W
            x_peak = max_idx % W

            # Grid coords
            yy = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, J, H, W)
            xx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, J, H, W)

            # Absolute distance to peak
            dist_y = torch.abs(yy - y_peak.view(B, J, 1, 1))
            dist_x = torch.abs(xx - x_peak.view(B, J, 1, 1))

            half_w = self.window_size // 2
            local_mask = (dist_y <= half_w) & (dist_x <= half_w)

            # Mask out non-local activations with large negative value
            masked_heatmaps = heatmaps.masked_fill(~local_mask, -1e9)
        else:
            masked_heatmaps = heatmaps

        # Flatten and compute softmax
        flat_masked = masked_heatmaps.view(B, J, -1)
        probs = F.softmax(flat_masked * self.temperature, dim=-1)
        probs = probs.view(B, J, H, W)

        # Compute expected x and y
        marginal_x = torch.sum(probs, dim=2)  # (B, J, W)
        marginal_y = torch.sum(probs, dim=3)  # (B, J, H)

        expected_x = torch.sum(marginal_x * self.grid_x, dim=2)  # (B, J)
        expected_y = torch.sum(marginal_y * self.grid_y, dim=2)  # (B, J)

        return torch.stack([expected_x, expected_y], dim=-1)
