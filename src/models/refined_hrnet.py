import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple, Optional, cast
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

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # Extract sub-configs
        if "model" in config:
            model_cfg: Dict[str, Any] = config["model"]
            hrnet_cfg: Dict[str, Any] = model_cfg.get("hrnet", config)
        else:
            hrnet_cfg = config

        self.hrnet = HRNet(hrnet_cfg)
        self.soft_argmax = SoftArgmax2D(temperature=100.0)
        self.refiner = PoseRefinementGCN(
            num_joints=int(hrnet_cfg.get("num_joints", 14)),
            hidden_dim=int(hrnet_cfg.get("gcn_hidden_dim", 64)),
        )

    @property
    def in_channels(self) -> int:
        return self.hrnet.in_channels

    @property
    def output_type(self) -> str:
        # We still primarily output heatmaps for the trainer
        return "heatmap"

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return_refined = kwargs.get("return_refined", False)
        heatmaps = cast(torch.Tensor, self.hrnet(x))

        # Initial coordinates from heatmaps
        coords = self.soft_argmax(heatmaps)  # (B, 14, 2)

        # Refine via GCN
        refined_coords = self.refiner(coords)  # (B, 14, 2)

        if return_refined:
            return heatmaps, refined_coords

        return heatmaps
