# State Tracker

- **Current Loop**: 20
- **Phase**: Brainstorming (Phase 1)
- **Status**: LOOP 19 COMPLETED. Skeleton collapse observed in anatomical constraint run.
- **Absolute Priority**: 
  1. **Fix Convergence**: Identify why normalized anatomical loss causes skeleton collapse (likely joint-coalescence local minima).
  2. **Baseline Recovery**: Recover 74%+ PCK by refining uncertainty weighting or increasing heatmap supervision.
- **Baseline**: Loop 17 (74.2% PCK).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.5 (corrected) | MPJPE (corrected) | Status |
|-----|---------|--------------------|--------------------|--------|
| loop17_uncertainty | soft-argmax | **74.2%** | 24.7 px | **SUCCESS** |
| loop9_anatomical_hinge | argmax | 73.0% | 27.4 px | RELIABLE |
| loop14_integral_regression | argmax | 30.5% | 69.3 px | FAILURE |
| loop15_occlusion_aware_integral | argmax | 31.9% | 63.6 px | FAILURE |
| loop16_sigma_curriculum | soft-argmax | 33.9% | 57.4 px | FAILURE |

The loop16 `best_model.pth` was saved based on **combined val loss** (heatmap MSE + coord L1 + anatomical), NOT on PCK. Combined loss is dominated by the anatomical term (lambda=0.5) and does not align with PCK. The actual best PCK epoch for loop16 is unknown because only epoch_1.pth (corrupted) and best_model.pth were downloaded.

## ⚠️ CRITICAL: Prerequisite Issue Before Next Loop

The training pipeline has a **fundamental loss-metric alignment problem** that must be resolved BEFORE continuing the autoresearch loop:

**Problem**: The combined training loss `L = MSE_heatmap + λ_coord * L1_coord + λ_ana * L_anatomical` does not monotonically correlate with val PCK. The auxiliary terms (anatomical, coordinate regression) operate at different scales and can dominate the loss landscape, causing `best_model.pth` to capture the epoch with the best auxiliary constraint satisfaction — not the epoch with the best pose accuracy.

**Consequence**: Comparing run results is unreliable; a "new best" checkpoint may actually be a worse predictor.

**Fix required (Phase 0 of next loop)**:
1. Normalize all auxiliary loss terms so they are dimensionless / same scale as heatmap MSE.
2. OR: restructure losses with adaptive weighting (e.g., uncertainty weighting by Kendall et al.).
3. OR: simplify — remove auxiliary losses that aren't clearly helping (the Graveyard shows anatomical loss rarely helps), and train clean baselines.

The `best_model.pth` saving criterion has been fixed to use **val PCK** (implemented 2026-05-11 in `base_trainer.py`). This fix takes effect from the next training run onward.

## Infrastructure Fixes Applied (2026-05-12)

- **API Stabilization**: Added missing `typing` imports (`Dict`, `Any`) to `src/api/main.py`. Fixed `NameError` on startup.
- **Evaluation Reporting**: Implemented `format_evaluation_metrics` helper in API to correctly parse per-joint PCK/MPJPE from `evaluation.json`, restoring dashboard plots.
- **History Persistence**: Updated `TrainingManager.get_status` to proactively reload `history.json` from disk, preventing "disappearing history" after training completion.
- **Evaluation Script**: Patched `scripts/evaluate.py` to resolve absolute/relative path mismatches when generating visual audit plots.
- **Environment**: Enforced `.venv\Scripts\python.exe` for all backend services to ensure dependency consistency.

## Iteration Log

| Loop ID | Hypothesis | Result | Corrected PCK | Action |
|---------|-----------|--------|--------------|--------|
| 3 | Improved Thermal + Full Dataset | SUCCESS | N/A (pre-audit) | Baseline established |
| 7 | Anatomical V1 (MSE length) | FAILURE | N/A | Fixed-length MSE over-regularizes |
| 8 | Anatomical Curriculum Warmup | FAILURE | N/A | Improves over 7 but below baseline |
| 9 | Foreshortening Hinge Loss | SUCCESS | ~78% (cover1+2, vis==0) | Best clean model to date |
| 10 | Angle Constraints (2D Hinge) | FAILURE | N/A | 2D angle != 3D ROM |
| 11 | Joint-Specific Weighting | FAILURE | N/A | Destabilized core structure |
| 12 | Occlusion Consistency Reg. | FAILURE | N/A | No extremity improvement |
| 13 | Multi-Scale Heatmap Supervision | FAILURE | N/A | Gradient noise |
| 14 | Soft-Argmax Integral Regression | FAILURE | 30.5% | Loss imbalance (aux term dominated) |
| 15 | Occlusion-Aware Integral Reg. | FAILURE | 31.9% | Loss imbalance (aux term dominated) |
| 16 | Adaptive Sigma Curriculum | FAILURE | 33.9% | Loss imbalance (aux term dominated) |
| 17 | Multi-Task Uncertainty Weighting | SUCCESS | **74.2%** | Kendall et al. weighting; PCK-aligned |
| 18 | GCN-based Pose Refinement | FAILURE | ~42% | GCN layer over-regularized prediction |
| 19 | Normalized Anatomical Constraints | FAILURE | 39.0% | **SKELETON COLLAPSE**: Joints coalesced to minimize length error |
