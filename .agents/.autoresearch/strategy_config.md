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
- **Remote Training Paths**: Kaggle GPU instances require absolute paths (e.g., `/root/project/data/raw`) for `data_root` to correctly initialize the `VIPCupDataset`.
- **Coordinate Transformations**: PIL `Image.rotate` is CCW. In `src/data/augmentations.py`, the joint transformation must use `new_x = x*cos + y*sin` and `new_y = -x*sin + y*cos` to match CCW image rotation in a y-down coordinate system.
- **Validation set**: Subjects 81-90 are used for "covered" modality validation (testing) while training is on subjects 1-80 (uncovered).
- **Model**: HRNet-W32 (W32_256x256).
