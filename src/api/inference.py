import torch
import os
from src.utils.pose import decode_heatmaps


class InferenceService:
    """
    Singleton service for high-performance, thread-safe inference.
    Decouples model loading and state management from the API workers.
    """

    _instance = None
    _model = None
    _device = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def load_model(self, checkpoint_path: str, force_reload: bool = False):
        if self._model is not None and not force_reload:
            return

        print(f"[InferenceService] Loading checkpoint: {checkpoint_path}")

        from src.models import load_model_for_inference
        from src.utils import load_config

        # We try to find the config for this run to ensure proper architecture
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        frozen_cfg_path = os.path.join(run_dir, "frozen_config.json")

        config = None
        if os.path.exists(frozen_cfg_path):
            import json

            with open(frozen_cfg_path, "r") as f:
                config = json.load(f)

        # Load model using the centralized utility
        self._model = load_model_for_inference(
            checkpoint_path, self._device, config=config
        )
        self._config = config or load_config()  # Fallback for metadata lookup

        print(f"[InferenceService] Model loaded successfully on {self._device}")

    @torch.no_grad()
    def predict(
        self, image_tensor: torch.Tensor, decode_method: str = "argmax"
    ) -> torch.Tensor:
        """
        Perform inference on a pre-processed image tensor.
        image_tensor: (1, C, H, W)
        Returns: (1, 14, 2) keypoints in image space
        """
        if self._model is None:
            raise RuntimeError("Model not loaded in InferenceService")

        image_tensor = image_tensor.to(self._device)
        outputs = self._model(image_tensor)

        if (
            hasattr(self._model, "output_type")
            and self._model.output_type == "coordinates"
        ):
            keypoints = outputs
        else:
            # Decode heatmaps using our optimized tensor-native implementation
            image_size = self._config.get("dataset", {}).get("image_size", [256, 256])
            keypoints = decode_heatmaps(
                outputs,
                image_size=tuple(image_size),
                method=decode_method,
                temperature=10.0,  # Default
            )

        return keypoints


# Global instance
inference_service = InferenceService()
