# Architecture Guide

## Model Overview: HRNet-W32
The project uses the High-Resolution Net (HRNet) backbone, which is specifically designed for tasks like human pose estimation where maintaining high-resolution representations throughout the network is critical.

### Key Components
- **Stems**: Initial stride-2 convolutions to reduce resolution to 1/4 size.
- **Parallel Streams**: Multiple branches processed simultaneously at different resolutions (1x, 2x, 4x, 8x downsampling).
- **Repeated Multi-Resolution Fusions**: Continuous exchange of information across branches to maintain rich spatial and semantic features.
- **Output Head**: 1x1 convolution projecting the high-resolution features (W32) to heatmap representations for 14 joints.

## Heatmap Regression
- Target: Gaussian heatmaps centered at normalized joint coordinates.
- Resolution: Heatmaps are generated at 1/4 the input resolution (e.g., 64x64 for 256x256 input).

## Modality Integration
- The model is designed to be modality-agnostic or multi-modal (input channels adjustable via config).
- Current configuration supports **RGB** and **IR** (LWIR).
    
## Inference Architecture
- **Self-Contained Decoding**: Models are wrapped in a `PoseDecodingWrapper` during inference. This wrapper encapsulates the heatmap-to-coordinate transformation (decoding), ensuring consistent behavior across different architectures and environments.
- **Decoding Configuration**: Every checkpoint (`.pth`) stores its optimal `decoding_config` (method, temperature, image size).
- **Loading**: The `src.models.load_model_for_inference` helper automatically detects the configuration and applies the wrapper.
- **Workflow**: Image Upload -> Preprocess -> `InferenceService` -> Wrapped Model Forward (direct coordinates) -> Coordinate Rescaling -> JSON Response.
- **Security Constraints**: `InferenceService` handles PyTorch 2.6+ `weights_only=True` restrictions by utilizing the unified `load_model_for_inference` helper with appropriate trust flags.

## Training Pipeline
- **Unified Entry Point**: `scripts/train.py` handles all training setups via a polymorphic factory.
- **Trainer Factory**: `src/training/factory.py` initializes the appropriate trainer based on configuration (Standard vs. UDA).
- **Supported Methods**:
    - **Standard Supervised**: MSE loss on heatmaps with optional anatomical constraints.
    - **Unsupervised Domain Adaptation (UDA)**: Uses a Domain Discriminator and Gradient Reversal Layer (GRL) to align feature distributions between clean (Source) and occluded (Target) IR images.
- **Orchestration**: `src/training/manager.py` manages training processes, supporting both local execution and remote GPU sessions (e.g., Kaggle) via SSH/Cloudflare tunnels.
