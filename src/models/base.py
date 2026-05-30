"""
Abstract base classes and interfaces for pose estimation models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all pose estimation models.
    Ensures a consistent interface for future architectures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the model with a configuration dictionary.

        Args:
            config: Model configuration parameters.
        """
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def output_type(self) -> str:
        """
        Returns the type of output the model produces.
        Possible values: "heatmap", "coordinates"
        """
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            If output_type is "heatmap": Heatmaps of shape (B, num_joints, H/k, W/k)
            If output_type is "coordinates": Joint coordinates of shape (B, num_joints, 2)
        """
        pass
