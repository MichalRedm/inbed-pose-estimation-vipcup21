# Data Pipeline Guide

## Dataset: SLP (Simultaneously-collected Multimodal Lying Pose)
The dataset consists of multi-modal images (RGB, LWIR, Depth) of subjects in various lying poses, both uncovered and covered by different types of blankets.

### Structure
- **Subjects**: 70 total.
- **Train Set**: Subjects 1-30 (Labeled).
- **Domain Adaptation**: Subjects 31-70 (Unlabeled/Covered).
- **Modalities**: RGB, IR, Depth.

### Preprocessing Requirements
- **Coordinate Shift**: Essential -1 pixel shift on x and y coordinates from raw MAT annotations to align with image coordinates.
- **Normalization**: Images normalized to [0, 1] range.
- **Resizing**: Default input size is 256x256. Joints must be scaled proportionally.
- **Homography Alignment**: IR and RGB modalities may require homography transformation for pixel-perfect alignment.

## Annotation Format
Raw annotations are stored in `joints_gt_<modality>.mat` files within each subject's directory.
- `joints_gt`: Matrix of shape `(3, 14, N)` where indices are `[coord, joint, img_idx]`.
- Coords: `0` (x), `1` (y), `2` (visibility).
- Visibility: `1` (Visible), `0` (Occluded).
