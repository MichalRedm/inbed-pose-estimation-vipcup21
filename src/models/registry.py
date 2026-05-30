from typing import Dict, Type, Callable, TypeVar
import torch.nn as nn

# Model Registry to allow easy switching between architectures
# Moving to a dedicated file prevents circular imports and registry isolation issues.
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

T = TypeVar("T", bound=Type[nn.Module])


def register_model(name: str) -> Callable[[T], T]:
    """Decorator to register a model class."""

    def decorator(cls: T) -> T:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator
