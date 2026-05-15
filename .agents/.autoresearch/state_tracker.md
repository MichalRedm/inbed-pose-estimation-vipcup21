# State Tracker

- **Current Loop**: 23
- **Phase**: Planning (Phase 2)
- **Status**: Loop 23 (Stabilized Pre-training + Selective Freezing) planned.
- **Absolute Priority**: 
  1. **Stabilize Pre-training**: Use layer freezing (Stem + Stage 1) to prevent feature washout.
  2. **Baseline Recovery**: Recover 46.6%+ PCK@0.2 by using Argmax + Pre-trained weights.
- **Baseline**: Loop 2 (46.6% PCK@0.2).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.2 (strict) | MPJPE | Status |
|-----|---------|--------------------|--------------------|--------|
| loop2_fixed_aug | argmax | **46.6%** | 29.6 px | **TOP PRECISION** |
| loop9_anatomical_hinge | soft-argmax | **45.1%** | 25.3 px | RELIABLE |
| loop3_improved_thermal | argmax | 44.7% | 27.4 px | SUCCESS |
| loop7_anatomical_v2 | argmax | 44.1% | 36.0 px | STABLE |
| loop17_uncertainty | soft-argmax | **43.1%** | 24.5 px | **TOP ACCURACY** |
| loop5_uda_refined | argmax | 36.2% | 31.4 px | UDA REF |
| loop4_uda_alignment | argmax | 35.6% | 40.3 px | UDA BASE |
| loop18_gcn_final_v5 | soft-argmax | 33.4% | 26.6 px | SUCCESS |
| loop19 | soft-argmax | 12.7% | 66.7 px | **SKELETON COLLAPSE** |

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
- **Corruption Recovery**: Restored `loop19` and `loop20` by recovering weights from `latest_model.pth` after `best_model.pth` was corrupted during save.
- **Remapping Fix**: Resolved a regression in `load_model_for_inference` where structural remapping (modules_list removal) failed for prefixed keys (e.g., `hrnet.`), which previously broke GCN-refined models like `loop18`.
- **Environment**: Enforced `.venv\Scripts\python.exe` for all backend services.

## Iteration Log

| Loop ID | Hypothesis | Result | Corrected PCK@0.2 | Action |
|---------|-----------|--------|--------------|--------|
| 1-8 | Initial Explorations | VARIOUS | N/A | **PURGED**: Legacy/Corrupted. |
| 2 | Baseline (Fixed Aug) | SUCCESS | **46.6%** | Solid baseline |
| 9 | Foreshortening Hinge Loss | SUCCESS | 45.1% | Best clean model to date |
| 17 | Multi-Task Uncertainty Weighting | SUCCESS | **43.1%** | Kendall et al. weighting; PCK-aligned |
| 18 | GCN-based Pose Refinement | SUCCESS | 33.4% | High accuracy, but GCN adds latency |
| 19 | Normalized Anatomical Constraints | FAILURE | 12.7% | **SKELETON COLLAPSE**: Joints coalesced |
| 20 | Pre-trained HRNet-W32 + Uncertainty Weighting | SUCCESS | 35.1% | **UNDERFITTING**: Linear loss trend; needs more epochs. |
| 21 | Enhanced HRNet (60 eps + Coord Loss + Conv1 Avg) | FINISHED | 32.0% | Model completed but PCK remained low. Stabilization fixes applied to infrastructure. |
| 22 | Pre-trained HRNet-W32 + Pure Heatmap MSE + Argmax | FINISHED | 32.5% | **STALLED**: Performance did not improve. Hypothesis: Feature washout from high-level gradients. |
| 23 | Stabilized Pre-training (Freeze Stem + Stage 1) | PLANNED | TBD | Focused fine-tuning of high-level pose logic. |

## Next Planned Steps
1.  **Implement Loop 23**: Update HRNet to support freezing and create config for 60-epoch run.
2.  **Monitor Convergence**: Verify if freezing Stem leads to faster, more stable PCK growth.
3.  **Visibility-Aware Attention**: Backlog.
