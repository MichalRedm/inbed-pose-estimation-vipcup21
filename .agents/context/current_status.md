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

- [x] **Evaluation Script Fixed**: `scripts/evaluate.py` now loads run-specific configs, auto-selects decoders, and uses the correct `vis<=1` mask.

### Priority 4 — HRNet Architecture
`src/models/hrnet.py` uses a simplified parallel-stream architecture. Full HRNet-W32 with feature pyramid fusion would improve representation quality.

## Next Steps (MANDATORY for next `/ml-autoresearch` session)

1. **Phase 0 (MANDATORY)**: **Re-evaluate all local runs**. Run `python scripts/evaluate.py --run_id [ID]` for Loop 9, 14, 15, and 16. Update `eval_framework.md` and `state_tracker.md` with the verified numbers.
2. **Phase 1 (MANDATORY)**: **Fix Loss-Metric Alignment**. Modify `StandardTrainer` to use uncertainty-based weighting or normalized auxiliary losses.
3. **Phase 2**: **Confirm Peak Performance**. Re-train Loop 16 with the fixed loss and PCK-based checkpointing.
