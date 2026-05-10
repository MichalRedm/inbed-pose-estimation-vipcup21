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
- **Training domain**: Uncovered images only (cover = "uncover"), subjects 1–80
- **Validation domain**: Covered images ONLY (cover1 + cover2), subjects 81–90 — this is the task objective
- **Evaluation**: Always use run's own `config.json`, not global `load_config()`. Always use vis≤1 mask.
- **Anatomical priors**: Must use normalized [0,1] coordinate space if implementing coordinate-space constraints
- **Remote GPU**: Use Kaggle T4 for training (40s/epoch typical). Local CPU only for quick eval/debug.

## Discovered Mechanics & Quirks

### Loss Architecture
- **Combined loss** = `MSE_heatmap + λ_coord * L1_coord_vis + λ_coord_occ * L1_coord_occ + λ_ana * L_anatomical`
- **⚠️ CRITICAL**: The combined loss does NOT reliably correlate with val PCK. The anatomical term (λ=0.5) and coordinate term dominate at different scales than the heatmap MSE (~0.002). This means `best_model.pth` saved by minimum combined loss may NOT be the best-PCK model.
- **Fix applied**: `best_model.pth` now saved by maximum val PCK (see `base_trainer.compute_val_pck()`). But the root problem (loss imbalance) remains and should be resolved in the next research loop.
- **Recommended next step**: Normalize auxiliary losses or use uncertainty-based multi-task weighting (Kendall et al., 2018).

### Heatmap Decoding
- **Argmax**: Correct for models trained with standard MSE heatmap loss (σ=2.0 constant). Produces quantized outputs at heatmap resolution / image scale. Identified by absence of `sigma_start`/`sigma_end` in training config.
- **Soft-argmax**: Correct for models trained with sigma curriculum (`sigma_start != sigma_end`). Requires sharp heatmap peaks to work reliably. If heatmaps have low signal-to-noise (early training, high σ), soft-argmax collapses to center prediction.
- **API auto-selection**: `src/api/main.py` reads run config.json and selects decoder automatically.
- **Trainer auto-selection**: `StandardTrainer.fit()` selects decoder for `compute_val_pck()` automatically.

### Sigma Curriculum
- In `StandardTrainer`, sigma decays linearly from `sigma_start` → `sigma_end` over 70% of training, then holds at `sigma_end`.
- Heatmap targets are regenerated per-batch during training (not precomputed in dataset) when `sigma_start != sigma_end`.
- Very small final sigma (e.g., σ=1.0 in 64×64 space) produces very sparse, peaked heatmaps — good for localization but sensitive to model convergence.

### Checkpointing
- `best_model.pth` checkpoint format: `{model_state_dict, config, best_val_pck, best_val_loss, optimizer_state_dict}`
- Per-epoch checkpoints: `epoch_N.pth` (same format)
- Remote training (Kaggle): historically only downloads `best_model.pth` — individual epoch checkpoints may be unavailable locally. `epoch_1.pth` for loop16 is corrupted (partial download).

### Data
- **Joint convention**: LSP 14-joint order. Index 0=R_Ankle ... 13=Head. Visibility: 0=visible, 1=occluded, 2=missing/OOB.
- **Coordinate convention**: GT joints stored as (x, y, vis) in ORIGINAL image space. Dataset scales them to `image_size` (256×256) in `__getitem__`.
- **Annotation offset**: Raw `.mat` files have 1-indexed coordinates; dataset subtracts 1 on load.

### Foreshortening
- 2D bone lengths are NOT fixed — they project from 3D. Use Hinge Loss (upper-bound only) for anatomical constraints, never fixed-length MSE.
- Curriculum warmup (~10 epochs) required before anatomical constraints to avoid local minima.

### Remote Training
- Kaggle T4: ~40s/epoch, 30-epoch run ~20min.
- Submission via `scripts/remote_train.py --run_id <id> --eval`.
- History/checkpoints downloaded automatically to `results/runs/<run_id>/`.
