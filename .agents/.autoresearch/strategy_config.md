# Strategy Config

## Repository Architecture Map
- `src/data/`: Dataset and Augmentations.
- `src/models/`: HRNet architecture and SoftArgmax layers.
- `src/training/`: Training logic (StandardTrainer, AnatomicalLoss).
- `scripts/`: Remote and local entry points.

## Hard Constraints
- Use remote GPU for training.
- Training set = Uncovered, Validation set = Covered.
- Anatomical priors MUST use normalized [0, 1] coordinate space.

## Discovered Mechanics & Quirks
- **Foreshortening**: 2D projection bone lengths are variable. Use Hinge Loss (Upper Bound) instead of MSE to avoid accuracy penalties on non-planar poses.
- **Curriculum**: Structural constraints need a ~10 epoch warmup to avoid gradient dominance in early training.
- **Remote Scaling**: Kaggle T4 GPUs are very fast (40s/epoch) for this dataset.
- **Checkpointing**: In curriculum training, `best_model.pth` might represent the unconstrained warmup phase. Always verify against `epoch_30.pth`.
