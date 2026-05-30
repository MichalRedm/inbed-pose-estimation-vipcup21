"""
Inference wrappers for making models self-contained.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union
from src.utils.pose import decode_heatmaps


class PoseDecodingWrapper(nn.Module):
    """
    Wraps a pose estimation model to handle heatmap decoding internally.
    This makes the model self-contained for inference, outputting joint coordinates.
    """

    def __init__(
        self,
        model: nn.Module,
        decode_method: str = "argmax",
        temperature: float = 10.0,
        image_size: Tuple[int, int] = (256, 256),
    ) -> None:
        """
        Initializes the wrapper.

        Args:
            model: The underlying pose estimation model.
            decode_method: Method used to decode heatmaps ('argmax' or 'soft-argmax').
            temperature: Temperature parameter for soft-argmax.
            image_size: Target image resolution for coordinate scaling.
        """
        super().__init__()
        self.model = model
        self.decode_method = decode_method
        self.temperature = temperature
        self.image_size = image_size

        # Determine if the underlying model outputs heatmaps that need decoding
        self._is_heatmap = (
            hasattr(model, "output_type") and getattr(model, "output_type") == "heatmap"
        )

        # If the model already outputs coordinates, the wrapper just passes them through.
        # This prevents "double-decoding" issues.

    @property
    def in_channels(self) -> int:
        """Returns the number of input channels expected by the backbone."""
        return int(getattr(self.model, "in_channels", 1))

    @property
    def output_type(self) -> str:
        """Returns the output type (always 'coordinates' for the wrapper)."""
        return "coordinates"

    def forward(
        self, x: torch.Tensor, return_heatmaps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Performs forward pass and decodes heatmaps if necessary.

        Args:
            x: Input image tensor.
            return_heatmaps: If True, also returns the raw heatmaps.

        Returns:
            Joint coordinates, or a tuple of (coordinates, heatmaps, ...)
        """
        outputs = self.model(x)

        if isinstance(outputs, tuple):
            heatmaps = outputs[0]
            extra = outputs[1:]
        else:
            heatmaps = outputs
            extra = ()

        if self._is_heatmap:
            # Safety check: if heatmaps has only 3 dims (J, H, W), add batch dim
            if heatmaps.dim() == 3:
                heatmaps = heatmaps.unsqueeze(0)

            joints = decode_heatmaps(
                heatmaps,
                self.image_size,
                method=self.decode_method,
                temperature=self.temperature,
            )
        else:
            joints = heatmaps

        if return_heatmaps:
            if extra:
                return (joints, heatmaps, *extra)
            return (joints, heatmaps)
        return joints
