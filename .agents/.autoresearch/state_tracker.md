# State Tracker

- **Current Loop**: 16 (UNDER RE-EVALUATION)
- **Phase**: Audit & Baseline Recovery
- **Status**: ALL PREVIOUS METRICS ARE SUSPECT. Discovered bugs in `evaluate.py` (wrong config, wrong mask, wrong decoder) and `BaseTrainer` (saved best by loss, not PCK). 
- **Absolute Priority**: Re-evaluate past runs and stabilize loss-metric alignment.
- **Baseline**: Loop 9 (Verified argmax) vs Loop 16 (Verified soft-argmax).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.5 (corrected) | MPJPE (corrected) |
|-----|---------|--------------------|--------------------|
| loop9_anatomical_hinge | argmax | ~73% (all covers) / ~78% (cover1+2 only, vis==0) | ~27 px |
| loop16_sigma_curriculum | soft-argmax | **78.8%** (cover1+2, vis≤1) | 26.4 px |

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
- **Evaluation**: `eval_framework.md` updated with corrected metric computation procedures.

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
| 14 | Soft-Argmax Integral Regression | SUCCESS | N/A (metrics suspect) | Sub-pixel accuracy claimed |
| 15 | Occlusion-Aware Integral Reg. | SUCCESS | N/A (metrics suspect) | Occluded joint supervision |
| 16 | Adaptive Sigma Curriculum | SUCCESS | **78.8%** (verified) | Sharp heatmaps via σ decay |
