# Data Pipeline Guide

## Dataset: SLP (Simultaneously-collected Multimodal Lying Pose)
The dataset consists of multi-modal images (RGB, LWIR, Depth) of subjects in various lying poses, both uncovered and covered by different types of blankets.

### Structure
- **Subjects**: 102 total (in full dataset).
- **Train Set (Source Domain)**: Subjects 1-30 (Uncovered images, **Annotated**).
- **Train Set (Target Domain)**: Subjects 31-80 (Covered images: cover1/cover2, **Unannotated**).
- **Validation/Test Set**: Subjects 81-90 (Covered images: cover1/cover2, **Annotated**).
- **Modalities**: RGB, IR (Primary focus).

### Preprocessing & Augmentation
- **Coordinate Shift**: Essential -1 pixel shift on x and y coordinates from raw MAT annotations.
- **Resizing**: Default input size is 256x256. Joints are scaled proportionally.
- **Thermal Diffusion Augmentation**: Simulates blankets in IR by applying localized Gaussian blur and intensity reduction (0.5-0.8) to lower-body regions based on joint visibility.
- **Geometric Augmentations**: Random flipping, rotation (±30°), and scaling (0.8-1.2x). **Warning**: Rotation must strictly match the CCW direction in `src/data/augmentations.py`.

## Annotation Format
Raw annotations are stored in `joints_gt_<modality>.mat` files.
- `joints_gt`: Matrix of shape `(3, 14, N)`.
- Coords: `0` (x), `1` (y), `2` (visibility).
- Joint Order (LSP): 0: R Ankle, 1: R Knee, 2: R Hip, 3: L Hip, 4: L Knee, 5: L Ankle, 6: R Wrist, 7: R Elbow, 8: R Shoulder, 9: L Shoulder, 10: L Elbow, 11: L Wrist, 12: Neck, 13: Head.

## Small Data & Modality Gap Challenges
- **Scarcity**: The source training domain consists of only ~1,350 annotated images. Training complex architectures (like ViTs) from scratch is mathematically prone to catastrophic overfitting (e.g., scoring <45% PCK).
- **The Gap**: Most pre-trained weights are RGB, but our target is Thermal (IR).
- **Required Mitigation**: Pre-trained weights are **mandatory**. To bridge the RGB-to-Thermal gap without washing out pre-trained structural priors, the pipeline must employ **Channel Replication** (copying IR to 3 channels) combined with **Discriminative Fine-Tuning** (freezing the backbone initially, then unfreezing with a very low learning rate like `1e-5`).
