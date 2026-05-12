import torch.nn as nn
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
        image_size: tuple = (256, 256),
    ):
        super().__init__()
        self.model = model
        self.decode_method = decode_method
        self.temperature = temperature
        self.image_size = image_size

        # Preserve the output_type of the original model if it's already coordinate-based
        # but the wrapper's purpose is to ensure it is coordinate-based.
        self._is_heatmap = (
            hasattr(model, "output_type") and model.output_type == "heatmap"
        )

    @property
    def output_type(self) -> str:
        return "coordinates"

    def forward(self, x, return_heatmaps=False):
        outputs = self.model(x)

        if isinstance(outputs, tuple):
            heatmaps = outputs[0]
            extra = outputs[1:]
        else:
            heatmaps = outputs
            extra = ()

        if self._is_heatmap:
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
                return joints, heatmaps, *extra
            return joints, heatmaps
        return joints
