from typing import Dict, Any, Type, Optional
import torch
import torch.nn as nn
from .wrapper import PoseDecodingWrapper

# Model Registry to allow easy switching between architectures
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls: Type[nn.Module]):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a model based on the configuration.

    Args:
        config: Full project configuration dictionary.

    Returns:
        An instantiated PyTorch model.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name")

    if not name:
        raise ValueError("Model name not specified in configuration ('model.name').")

    if name not in MODEL_REGISTRY:
        # Import all modules in the package to ensure registration
        from . import hrnet  # noqa: F401
        from . import refined_hrnet  # noqa: F401

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

    # Get specific model configuration (e.g., model.hrnet)
    specific_cfg = model_cfg.get(name, {})

    # Instantiate and return the model
    return MODEL_REGISTRY[name](specific_cfg)


def load_model_for_inference(
    checkpoint_path: str, device: torch.device, config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Loads a model from a checkpoint and automatically wraps it in PoseDecodingWrapper
    if it's a heatmap-based model.
    """
    # Load state. We set weights_only=False because older checkpoints may contain
    # numpy scalars or other legacy objects that PyTorch 2.6+ blocks by default.
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use config from checkpoint if available, else provided config
    if isinstance(state, dict) and "config" in state:
        model_config = state["config"]
    elif config:
        model_config = config
    else:
        from src.utils import load_config

        model_config = load_config()

    # Build base model
    model = build_model(model_config).to(device)

    # Load state dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        # Fallback for old checkpoints that saved state_dict directly
        # Filter out metadata keys to avoid RuntimeError
        metadata_keys = ["decoding_config", "best_optimized_pck", "config"]
        filtered_state = {k: v for k, v in state.items() if k not in metadata_keys}
        model.load_state_dict(filtered_state, strict=False)

    # Apply wrapper if it's a heatmap model
    if hasattr(model, "output_type") and model.output_type == "heatmap":
        decoding_cfg = {}
        if isinstance(state, dict) and "decoding_config" in state:
            decoding_cfg = state["decoding_config"]
        else:
            # Fallback to defaults or config
            train_cfg = model_config.get("training", {})
            decoding_cfg = {
                "method": train_cfg.get("decode_method", "argmax"),
                "temperature": train_cfg.get("decode_temperature", 10.0),
                "image_size": model_config.get("dataset", {}).get(
                    "image_size", [256, 256]
                ),
            }

        model = PoseDecodingWrapper(
            model,
            decode_method=decoding_cfg.get("method", "argmax"),
            temperature=decoding_cfg.get("temperature", 10.0),
            image_size=tuple(decoding_cfg.get("image_size", [256, 256])),
        )

    model.eval()
    return model
