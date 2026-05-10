# Current Status & Roadmap

## Implemented Features
- [x] **Project Infrastructure**: Relocated `remote_gpu.py` to `src/utils/`, initialized `src` packages, set up `.venv` / `.env`.
- [x] **Data Loading**: `VIPCupDataset` with cover-aware splits, coordinate shifting, modality support.
- [x] **Kaggle Integration**: `scripts/download_dataset.py`, `scripts/remote_train.py` for Kaggle T4 training.
- [x] **Model Architecture**: HRNet-W32 in `src/models/hrnet.py` with `SoftArgmax2D` layer.
- [x] **Training Pipeline**: `StandardTrainer` with heatmap MSE, optional anatomical hinge loss, optional coordinate regression loss, sigma curriculum.
- [x] **Inference API**: FastAPI server (`src/api/main.py`) with per-run model caching and auto-selected decoder (argmax vs soft-argmax).
- [x] **Dashboard**: React/Vite UI; canvas-based inference overlay (pixel-accurate skeleton); global model selector; evaluation page.
- [x] **Metrics**: PCK@0.5 (torso-relative) and MPJPE — standardized in `src/utils/pose.py`.
- [x] **PCK-Based Checkpointing**: `BaseTrainer.compute_val_pck()` computes true PCK each epoch; `best_model.pth` saved at epoch of highest val PCK (not lowest combined loss).
- [x] **Soft-Argmax Decoding**: Sub-pixel coordinate decoding in API and training evaluation; auto-selected per run.

## Current Technical Debts / Open Issues

### ⚠️ Priority 1 — Loss-Metric Alignment (BLOCKER for next research loop)
The combined training loss (`MSE_heatmap + λ_coord * L1 + λ_ana * L_hinge`) does not reliably correlate with val PCK. Auxiliary terms operate at different scales and can dominate the loss landscape. This caused `best_model.pth` for previous runs to capture the wrong epoch. **Must be resolved before trusting any future A/B comparisons.**
- **Fix direction**: Uncertainty-based multi-task loss weighting (Kendall et al., 2018), or normalize auxiliary losses to the scale of heatmap MSE.

### ⚠️ Priority 2 — Corrected Baseline Metrics Needed
All PCK figures reported from Loops 9–15 were produced by a flawed `scripts/evaluate.py` (used global config, wrong visibility mask, hardcoded soft-argmax). Verified baselines:
- **Loop 9 (argmax, cover1+2, vis≤1)**: ~78% PCK — needs exact measurement
- **Loop 16 (soft-argmax, cover1+2, vis≤1)**: **78.8% PCK, 26.4px MPJPE** — verified

### Priority 3 — `scripts/evaluate.py` Bug
This script loads the global default config instead of the run-specific config. It should be refactored to accept `--run_id` and load `results/runs/<run_id>/config.json` automatically.

### Priority 4 — HRNet Architecture
`src/models/hrnet.py` uses a simplified parallel-stream architecture. Full HRNet-W32 with feature pyramid fusion would improve representation quality.

## Next Steps (for next `/ml-autoresearch` session)

1. **Phase 0**: Read all `.agents/.autoresearch/` files. Note the PREREQUISITE flag on Hypothesis #1.
2. **Phase 1**: Implement loss-metric alignment (uncertainty weighting or loss normalization). Run loop17 with this fix and verify that val_pck trajectory in `history.json` now matches intuitive training progress.
3. **Phase 2**: Once loss is reliable, re-establish a clean baseline by re-running loop9 or loop16 with the corrected saving criterion to get a solid reference point.
4. **Phase 3+**: Continue with architecture / augmentation improvements using reliable metrics.
