"""
Data augmentation package for multi-modal pose estimation.
Includes geometric, intensity, thermal diffusion, and GAN-based domain translation augmentations.
"""

from .augmenter import (
    DataAugmenter,
    get_available_augmentations,
    apply_custom_augmentations,
)

# Export individual classes for direct use if needed
from .geometric import HorizontalFlipAugmentation, AffineAugmentation
from .intensity import ThermalIntensityJitter, IRSensorNoise
from .thermal import ThermalDiffusionAugmenter, AdvancedCoverAugmenter
from .occlusion import CutoutAugmentation
from .domain import CycleGANAugmentation, CUTAugmentation

__all__ = [
    "DataAugmenter",
    "get_available_augmentations",
    "apply_custom_augmentations",
    "HorizontalFlipAugmentation",
    "AffineAugmentation",
    "ThermalIntensityJitter",
    "IRSensorNoise",
    "ThermalDiffusionAugmenter",
    "AdvancedCoverAugmenter",
    "CutoutAugmentation",
    "CycleGANAugmentation",
    "CUTAugmentation",
]
