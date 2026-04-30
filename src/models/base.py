import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all pose estimation models.
    Ensures a consistent interface for future architectures.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Heatmaps of shape (B, num_joints, H/4, W/4)
        """
        pass
