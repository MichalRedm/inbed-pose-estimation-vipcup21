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
| loop17_uncertainty | soft-argmax | **75.1%** | 25.8 px | **SUCCESS** |
| loop18_gcn_final_v5 | soft-argmax | **75.1%** | 26.6 px | SUCCESS |
| loop9_anatomical_hinge | soft-argmax | 73.0% | 27.4 px | RELIABLE |
| loop3_improved_thermal | argmax | 72.9% | 27.4 px | SUCCESS |
| loop2_fixed_aug | argmax | 71.3% | 29.6 px | SUCCESS |
| loop7_anatomical_v2 | argmax | 70.1% | 36.0 px | STABLE |
| loop4_uda_alignment | argmax | 62.6% | 40.3 px | UDA BASE |
| loop5_uda_refined | argmax | 61.6% | 40.5 px | UDA REF |
| loop19 | soft-argmax | 36.6% | 67.8 px | **SKELETON COLLAPSE** |

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

- **Model Self-Containment**: Standardized all legacy checkpoints to include weights, config, and decoding settings in a single `.pth` file.
- **Decoding Autopilot**: Implemented `PoseDecodingWrapper` to automatically apply the best decoding method (argmax vs soft-argmax) at inference time based on embedded metadata.
- **API Stabilization**: Resolved `NameError` and type-hinting issues in `src/api/main.py`. Ensured robust loading of historical checkpoints.
- **Corruption Recovery**: Restored `loop19` by recovering weights from `latest_model.pth` after `best_model.pth` was corrupted.
- **Cleanup**: Purged corrupted and legacy runs (`loop1`, `loop19_auto_eval`, `loop5_test`, `loop4_telemetry_check`) to stabilize dashboard and evaluation pipelines.
- **Environment**: Enforced `.venv\Scripts\python.exe` for all backend services to ensure dependency consistency.

## Iteration Log

| Loop ID | Hypothesis | Result | Corrected PCK | Action |
|---------|-----------|--------|--------------|--------|
| 1-8 | Initial Explorations | VARIOUS | N/A | **PURGED**: Legacy/Corrupted. |
| 9 | Foreshortening Hinge Loss | SUCCESS | 73.0% | Best clean model to date |
| 10-16 | Auxiliary Loss Experiments | FAILURE | <35% | Loss imbalance (aux term dominated) |
| 17 | Multi-Task Uncertainty Weighting | SUCCESS | **75.1%** | Kendall et al. weighting; PCK-aligned |
| 18 | GCN-based Pose Refinement | SUCCESS | 75.1% | High accuracy, but GCN adds latency |
| 19 | Normalized Anatomical Constraints | FAILURE | 36.6% | **SKELETON COLLAPSE**: Joints coalesced |
