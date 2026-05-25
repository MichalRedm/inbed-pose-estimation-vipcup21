# Strategy Config

## Repository Architecture Map
- `src/data/`: `VIPCupDataset` (IR/RGB, cover-aware), `DataAugmenter` (flip, rotate, scale, occlusion)
- `src/models/`: HRNet-W32 (`hrnet.py`), `SoftArgmax2D` layer (`layers.py`), model factory (`__init__.py`)
- `src/training/`: `BaseTrainer` (abstract, PCK-based checkpointing), `StandardTrainer` (heatmap MSE + optional anatomical + coord regression), `AnatomicalLoss` (`losses.py`)
- `src/utils/`: `decode_heatmaps()` (argmax + soft-argmax), `compute_pck()`, `compute_mpjpe()`, `LSP_JOINT_NAMES`
- `src/api/main.py`: FastAPI inference server; auto-selects decoder per run; caches loaded models by run_id key
- `scripts/`: Remote and local entry points. `evaluate.py` is fixed and recommended for post-training validation.
- `dashboard/`: React/Vite frontend; inference uses canvas-based overlay; model selected globally via header
- `results/runs/<run_id>/`: `config.json` (run-specific config), `checkpoints/best_model.pth`, `history.json`
- `.agents/`: Agent context, autoresearch state, architecture docs

## Hard Constraints
- **Training Source domain**: Uncovered images ONLY, subjects 1–30 (Annotated).
- **Training Target domain**: Covered images ONLY (cover1 + cover2), subjects 31–80 (Unannotated). *Note: Currently excluded by dataloader, but available for Semi-Supervised/UDA strategies.*
- **Validation domain**: Covered images ONLY (cover1 + cover2), subjects 81–90 — this is the task objective
- **Pre-training Mandate**: Due to small dataset size (~1,350 source images), models MUST utilize massive pre-trained weights (e.g., MS COCO for structural priors or Thermal-native datasets). Training from scratch is forbidden unless establishing a baseline.
- **Evaluation**: Always use run's own `config.json`, or the embedded `decoding_config` in the checkpoint. Threshold standard: **PCK@0.2**.
- **Anatomical priors**: Must use normalized [0,1] coordinate space if implementing coordinate-space constraints
- **Remote GPU**: Use Kaggle T4 for training (40s/epoch typical). Be sure to wait for the training to finish before proceeding to the next step.
- **Environment**: Use `.venv\Scripts\python.exe` for all local backend services (API, Training Manager) to ensure strict dependency parity.
- **Paths**: `scripts/evaluate.py` requires absolute paths for reliable visual audit plot generation.

## Discovered Mechanics & Quirks

### Loss Architecture
- **Combined loss** = `MSE_heatmap + λ_coord * L1_coord_vis + λ_coord_occ * L1_coord_occ + λ_ana * L_anatomical`
- **Fix applied**: `best_model.pth` now saved by maximum val PCK (see `base_trainer.compute_val_pck()`).
- **Update (2026-05-11)**: Implemented **Uncertainty-based Multi-task Weighting** (Kendall et al., 2018). The trainer now learns $\sigma_i$ for each loss term to automatically balance them. Enabled via `use_uncertainty_weighting: true` in config.

### Heatmap Decoding
- **Argmax**: Correct for models trained with standard MSE heatmap loss (σ=2.0 constant). Produces quantized outputs at heatmap resolution / image scale. Identified by absence of `sigma_start`/`sigma_end` in training config.
- **Soft-argmax**: Correct for models trained with sigma curriculum (`sigma_start != sigma_end`). Requires sharp heatmap peaks to work reliably. If heatmaps have low signal-to-noise (early training, high σ), soft-argmax collapses to center prediction.
- **API auto-selection**: `src/api/main.py` reads run config.json and selects decoder automatically.
- **API Metrics Formatting**: All evaluation metrics must pass through `format_evaluation_metrics()` in `src/api/main.py`. This normalizes `per_joint_mpjpe` and `per_joint_pck` into a unified `per_joint_metrics` list for the React dashboard charts.
- **Trainer auto-selection**: `StandardTrainer.fit()` selects decoder for `compute_val_pck()` automatically.

### Sigma Curriculum
- In `StandardTrainer`, sigma decays linearly from `sigma_start` → `sigma_end` over 70% of training, then holds at `sigma_end`.
- Heatmap targets are regenerated per-batch during training (not precomputed in dataset) when `sigma_start != sigma_end`.
- Very small final sigma (e.g., σ=1.0 in 64×64 space) produces very sparse, peaked heatmaps — good for localization but sensitive to model convergence.

### Checkpointing
- **Self-Contained Checkpoints (2026-05-12)**: `best_model.pth` now bundles `{model_state_dict, config, decoding_config, metrics}`.
- **Decoding Autopilot**: `src/models/__init__.py → load_model_for_inference` and `scripts/evaluate.py` automatically load `decoding_config` (method, temperature) from the checkpoint.
- Per-epoch checkpoints: `epoch_N.pth` (same format)
- Remote training (Kaggle): historically only downloads `best_model.pth`.

### Data
- **Joint convention**: LSP 14-joint order. Index 0=R_Ankle ... 13=Head. Visibility: 0=visible, 1=occluded, 2=missing/OOB.
- **Coordinate convention**: GT joints stored as (x, y, vis) in ORIGINAL image space. Dataset scales them to `image_size` (256×256) in `__getitem__`.
- **Annotation offset**: Raw `.mat` files have 1-indexed coordinates; dataset subtracts 1 on load.

### Foreshortening
- Foreshortening: 2D bone lengths are NOT fixed — they project from 3D. Use Hinge Loss (upper-bound only) for anatomical constraints, never fixed-length MSE.
- **Skeleton Collapse Risk**: In normalized [0,1] coordinate space, a strong anatomical hinge loss can drive the model toward a degenerate solution where all joints are at (0.5, 0.5), as this yields zero bone length error. Requires strong heatmap supervision and low initial $\lambda_{ana}$ to anchor the structure.
- Curriculum warmup (~10 epochs) required before anatomical constraints to avoid local minima.

### Remote Training
- Kaggle T4: ~40s/epoch, 30-epoch run ~20min.
- **MANDATORY**: Always launch training via the API `POST /training/start` with `"remote": true`.
- **FORBIDDEN**: Do not run `scripts/remote_train.py` directly from the terminal.
- History/checkpoints downloaded automatically to `results/runs/<run_id>/`.

### Baseline Verification
- All baselines must be re-evaluated locally using `scripts/evaluate.py` to ensure absolute parity.
- Suspect metrics from remote environments must be flagged in `state_tracker.md` and verified before being used for comparative research.
- Verification is a mandatory prerequisite for each new research phase.

### Keypoint Flipping Swap Mechanics (Discovered 2026-05-19)
- **Problem**: When a horizontal flip is applied to the image (with a 50% probability), coordinate symmetry requires that the keypoint semantic labels for the left and right sides are also swapped (e.g. swapping index of left knee and right knee). 
- **Fix**: Implemented keypoint indices reordering for horizontal flips in `src/data/augmentations.py`.

### Heatmap Curriculum & GPU Parallelization (Discovered 2026-05-19)
- **Problem**: Because standard PyTorch dataloaders spawn persistent worker processes that copy the dataset object state once at startup, any dynamic parameter modifications made on the main thread (like decaying `sigma` for curriculum learning) do NOT propagate to the dataloader workers. Workers remain stuck at `sigma_start` for the entire run.
- **Fix**: Heatmap target generation was moved out of CPU dataset loading completely and replaced with a vectorized GPU-based on-the-fly generator (`generate_pytorch_heatmaps` inside `src/training/standard_trainer.py`) to guarantee correct curriculum scheduling and high training speed.
