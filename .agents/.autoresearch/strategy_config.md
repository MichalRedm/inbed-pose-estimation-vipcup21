# Strategy Config

## Repository Architecture Map
- `src/data/`: Dataset and Augmentations.
- `src/models/`: HRNet architecture.
- `src/training/`: Training logic.
- `scripts/`: Entry points.

## Hard Constraints
- Use remote GPU for training.
- Inform user if remote GPU is unavailable.
- Training set = Uncovered, Validation set = Covered.

## Discovered Mechanics & Quirks
- Validation set is actually for testing (per user).
- Augmentations recently added but didn't help much.
- Dataset is SLP (Simultaneously-collected Multimodal Lying Pose).
- Model is HRNet-W32.
