from typing import Dict, Any, Type
import torch.nn as nn

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

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

    # Get specific model configuration (e.g., model.hrnet)
    specific_cfg = model_cfg.get(name, {})

    # Instantiate and return the model
    return MODEL_REGISTRY[name](specific_cfg)
