# State Tracker

- **Current Loop**: 17
- **Phase**: Loss Alignment Implementation
- **Status**: IMPLEMENTED. Added `UncertaintyWeighting` to `StandardTrainer`.
- **Absolute Priority**: 
  1. **Verification**: Explicitly verify if the evaluations of previous baselines (saved in the Iteration Log) are correct using the fixed local evaluation framework. This is a high-priority prerequisite for all comparative analysis.
  2. **Loop 17**: Validate that learned weights balance the loss and improve PCK correlation.
- **Baseline**: Loop 16 (Verified 78.8% PCK - Re-verification recommended upon structural change).

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

## Infrastructure Fixes Applied (2026-05-11)

- **Dashboard**: Canvas-based inference overlay fully working. Image sizing restored. Skeleton overlay pixel-accurate.
- **API**: Auto-selects `argmax` vs `soft-argmax` decoding per run based on `sigma_start`/`sigma_end` keys in run config.json.
- **Trainer**: `BaseTrainer.compute_val_pck()` added; `StandardTrainer.fit()` now saves `best_model.pth` based on highest val PCK rather than lowest combined loss.
- **Evaluation**: `eval_framework.md` updated with corrected metric computation procedures and Verification Protocol.

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
