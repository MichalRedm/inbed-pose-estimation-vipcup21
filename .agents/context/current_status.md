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
- [x] **Self-Training (Loop 54)**: EMA Teacher-Student pipeline with CUT strong augmentation. Achieved **82.5% PCK@0.2** (New all-time record).

## Current Technical Debts / Open Issues

### ⚠️ Priority 1 — Loss-Metric Alignment
The combined training loss (`MSE_heatmap + λ_coord * L1 + λ_ana * L_hinge`) does not reliably correlate with val PCK. Auxiliary terms operate at different scales and can dominate the loss landscape.
- **Update**: Uncertainty-based multi-task weighting is implemented and utilized in StandardTrainer. Loop 54 v3 confirmed stable behavior.

### 🏁 Milestones Reached
- [x] **Evaluation Script Fixed**: `scripts/evaluate.py` now loads run-specific configs, auto-selects decoders, and uses the correct `vis<=1` mask.
- [x] **Corrected Baselines Established**: Established strict PCK@0.2 baselines for Loop 44 (77.8%), Loop 53 (78.7%), and Loop 54 (82.5%).

### Priority 4 — HRNet Architecture
`src/models/hrnet.py` uses a simplified parallel-stream architecture. Full HRNet-W32 with feature pyramid fusion would improve representation quality.

## Next Steps (MANDATORY for next `/ml-autoresearch` session)

1. **Phase 0 (MANDATORY)**: **Re-evaluate all local runs**. Run `python scripts/evaluate.py --run_id [ID]` for Loop 9, 14, 15, and 16. Update `eval_framework.md` and `state_tracker.md` with the verified numbers.
2. **Phase 1 (MANDATORY)**: **Fix Loss-Metric Alignment**. Modify `StandardTrainer` to use uncertainty-based weighting or normalized auxiliary losses.
3. **Phase 2**: **Confirm Peak Performance**. Re-train Loop 16 with the fixed loss and PCK-based checkpointing.
