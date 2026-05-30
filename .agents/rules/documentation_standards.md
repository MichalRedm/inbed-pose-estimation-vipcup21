# Documentation Standards

This project follows strict Test-Driven Development (TDD) and Mypy-enforced type safety. To ensure high code quality without redundant maintenance, we use a specific documentation style.

## 1. Docstring Style: Google (No Types)

We use the **Google Python Style Guide** for docstrings, with one critical modification: **do not include type information** in the docstrings.

### Why No Types?
- **Mypy Enforcement**: All functions and methods must have type hints in the signature (`disallow_untyped_defs = True`).
- **Single Source of Truth**: Including types in docstrings creates redundancy. If types change, they must be updated in two places, leading to drift.
- **Readability**: Docstrings remain clean and focused on the *purpose* and *semantics* of parameters.

## 2. Format Requirements

### Module-Level Docstrings
Every Python file must start with a module-level docstring explaining its purpose.
```python
"""
This module handles Gaussian heatmap generation for joint targets.
It supports vectorized GPU rendering and dynamic sigma curriculum.
"""
```

### Class Docstrings
Classes should have a summary of their responsibility.
```python
class HeatmapGenerator(nn.Module):
    """
    Generates 2D Gaussian heatmaps from coordinate labels.
    
    Supports multi-scale targets and dynamic sigma scheduling.
    """
```

### Function/Method Docstrings
Functions should document their behavior, arguments, and return values.

```python
def generate_heatmaps(
    coords: torch.Tensor, 
    sigma: float, 
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Computes Gaussian heatmaps for a batch of coordinates.

    Args:
        coords: Tensor of shape (B, J, 2) in normalized [0, 1] space.
        sigma: Standard deviation of the Gaussian peak.
        image_size: Target resolution as (Height, Width).

    Returns:
        Tensor of shape (B, J, H, W) containing the heatmaps.
    """
```

## 3. Mandatory Sections
- **Summary**: A concise one-liner.
- **Args**: List each parameter with a brief description of its semantic meaning.
- **Returns**: Describe the return value (omit if `None` or a simple constructor).
- **Raises**: List exceptions that are explicitly raised by the function.
- **Note/Example**: Use these for complex logic or usage patterns.

## 4. Scope
- **All Public Symbols**: All modules, classes, and functions that are not prefixed with `_` must have docstrings.
- **Complex Private Symbols**: Internal logic that is non-trivial should also be documented.
