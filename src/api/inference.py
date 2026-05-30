import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional, Union, Tuple, List, cast
from src.utils.pose import decode_heatmaps


class InferenceService:
    """
    Singleton service for high-performance, thread-safe inference.
    Decouples model loading and state management from the API workers.
    """

    _instance: Optional["InferenceService"] = None
    _model: Optional[nn.Module] = None
    _device: torch.device
    _config: Optional[Dict[str, Any]] = None
    _current_checkpoint: Optional[str] = None

    def __new__(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            # Initialize _device on the instance
            cls._instance._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def load_model(self, checkpoint_path: str, force_reload: bool = False) -> None:
        if (
            self._model is not None
            and self._current_checkpoint == checkpoint_path
            and not force_reload
        ):
            return

        print(f"[InferenceService] Loading checkpoint: {checkpoint_path}")

        from src.models import load_model_for_inference
        from src.utils import load_config
        import shutil
        import uuid

        # SHADOW COPY: On Windows, loading directly can block the trainer from saving.
        # We copy to a temp location first.
        os.makedirs("scratch/inference_cache", exist_ok=True)
        shadow_path = os.path.join(
            "scratch/inference_cache", f"tmp_{uuid.uuid4().hex}.pth"
        )

        try:
            shutil.copy2(checkpoint_path, shadow_path)

            # We try to find the config for this run to ensure proper architecture
            run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            frozen_cfg_path = os.path.join(run_dir, "frozen_config.json")

            config: Optional[Dict[str, Any]] = None
            if os.path.exists(frozen_cfg_path):
                import json

                try:
                    with open(frozen_cfg_path, "r") as f:
                        config = json.load(f)
                except Exception as e:
                    print(
                        f"[InferenceService] Warning: Failed to load frozen config: {e}"
                    )

            # Load model using the centralized utility (now from the shadow copy)
            new_model = load_model_for_inference(
                shadow_path, self._device, config=config
            )

            # Atomic update
            self._model = new_model
            self._config = config or load_config()
            self._current_checkpoint = checkpoint_path
            print(f"[InferenceService] Model loaded successfully on {self._device}")

        except Exception as e:
            print(f"[InferenceService] Error loading model: {e}")
            raise e
        finally:
            # Clean up shadow copy immediately after loading
            if os.path.exists(shadow_path):
                try:
                    os.remove(shadow_path)
                except OSError:
                    pass

    @torch.no_grad()
    def predict(
        self,
        image_tensor: torch.Tensor,
        decode_method: str = "argmax",
        return_heatmaps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Perform inference on a pre-processed image tensor.
        Returns: (1, 14, 2) keypoints in image space.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded in InferenceService")

        image_tensor = image_tensor.to(self._device)

        # 1. Primary path: Model is self-contained (wrapped or native coordinate regression)
        if (
            hasattr(self._model, "output_type")
            and getattr(self._model, "output_type") == "coordinates"
        ):
            outputs = self._model(image_tensor, return_heatmaps=return_heatmaps)
            return cast(Union[torch.Tensor, Tuple[torch.Tensor, ...]], outputs)

        # 2. Legacy path: Model returns raw heatmaps, we decode manually
        outputs = self._model(image_tensor)

        # Determine image size for decoding
        dataset_cfg: Dict[str, Any] = self._config.get("dataset", {}) if self._config else {}
        image_size: List[int] = dataset_cfg.get("image_size", [256, 256])

        heatmaps = outputs[0] if isinstance(outputs, tuple) else outputs
        keypoints = decode_heatmaps(
            heatmaps,
            image_size=(image_size[0], image_size[1]),
            method=decode_method,
            temperature=10.0,
        )

        if return_heatmaps:
            return keypoints, heatmaps
        return keypoints


# Global instance
inference_service = InferenceService()
