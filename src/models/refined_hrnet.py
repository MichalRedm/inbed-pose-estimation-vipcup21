import torch.nn as nn
from .hrnet import HRNet
from .refinement import PoseRefinementGCN
from .layers import SoftArgmax2D
from .registry import register_model


@register_model("refined_hrnet")
class GCNRefinedHRNet(nn.Module):
    """
    HRNet backbone with a GCN refinement stage.
    Output heatmaps (standard) AND refined coordinates.
    """

    def __init__(self, config):
        super().__init__()
        # Extract sub-configs
        if "model" in config:
            model_cfg = config["model"]
            hrnet_cfg = model_cfg.get("hrnet", config)
        else:
            hrnet_cfg = config

        self.hrnet = HRNet(hrnet_cfg)
        soft_argmax_window = hrnet_cfg.get("soft_argmax_window", 15)
        self.soft_argmax = SoftArgmax2D(
            temperature=100.0, window_size=soft_argmax_window
        )
        self.refiner = PoseRefinementGCN(
            num_joints=hrnet_cfg.get("num_joints", 14),
            hidden_dim=hrnet_cfg.get("gcn_hidden_dim", 64),
        )

    @property
    def output_type(self) -> str:
        # We still primarily output heatmaps for the trainer
        return "heatmap"

    def forward(self, x, return_refined=False):
        heatmaps = self.hrnet(x)

        # Initial coordinates from heatmaps
        coords = self.soft_argmax(heatmaps)  # (B, 14, 2)

        # Refine via GCN
        refined_coords = self.refiner(coords)  # (B, 14, 2)

        if return_refined:
            return heatmaps, refined_coords

        return heatmaps
