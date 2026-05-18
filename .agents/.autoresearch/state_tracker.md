# State Tracker

- **Current Loop**: 26
- **Phase**: Phase 3: Implementation
- **Status**: Implementing the 5-component robust curriculum scratch pipeline (`feat/curriculum-robust-aug` branch) combining Sigma Curriculum (3.0→1.5), Structured Cutout, Thermal Intensity Jitter, Sensor Noise Injection, and Random Translation to directly bridge the newly discovered cover-type domain gap.
- **Absolute Priority**: 
  1. **PIVOT CONFIRMED**: Pretrained approach definitively abandoned. The HRNet RGB→IR domain gap, combined with the 80-subject training set scale and 1-channel conv1 limitation, creates a structural ceiling at ~42% that progressive unfreezing cannot breach.
  2. **Scratch Baselines remain superior**: loop2 (46.6%), loop9 (45.1%), loop17 (43.1%) are the targets.
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
| loop23_stabilized_pretraining | argmax | **41.0%** | 37.0 px | **SUCCESSFUL FINE-TUNING** |
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
| 23 | Stabilized Pre-training (Freeze Stem + Stage 1) | SUCCESS | **41.0%** | Successfully fine-tuned pre-trained backbone after resolving nested downsample/transition loading mismatch! |
| 24 | Discriminative LR (0.1× backbone, fully unfrozen) | UNDERPERFORMED | **36.7%** | Worse than Loop 23 (41.0%) despite discriminative LR. Root cause: initial gradient washout from random head is too severe even at 1e-5 backbone LR. Linear convergence visible (still improving at epoch 30). |
| 25 | Progressive Unfreezing (Phase 1: Frozen 15 ep, Phase 2: Disc. LR) | UNDERPERFORMED | **41.87%** (peak E33) / 40.33% final | Best pretrained result yet, but still 4.7pp below scratch baseline (46.6%). Hard plateau at ~42% despite 50 epochs and full backbone fine-tuning. Confirmed train-val divergence (gap grew 1731% in Phase 2), indicating mild overfitting on 80-subject set. **VERDICT: Structural ceiling on pretrained route. Pivot to scratch-based improvements.** |

## ⚠️ CRITICAL: Pretrained Route Post-Mortem (2026-05-18 — Final)

**Root causes for the confirmed structural ceiling at ~42% PCK:**

1. **RGB→IR Domain Gap is fundamental, not addressable by scheduling**: ImageNet RGB features (texture, color, spatial gradients) have minimal overlap with thermal IR (heat diffusion, emissivity, body-mass heat signatures). The backbone, despite 50 epochs of fine-tuning, retains structural biases that are misaligned with the thermal domain. This is consistent with the literature: domain-specific pre-training (e.g., grayscale-COCO or thermal-dataset pre-training) is required to bridge this gap reliably.
2. **1-channel conv1 adaptation is a known weak link**: HRNet-W32 conv1 is designed for 3-channel RGB. Averaging the 3 input channels to adapt to 1-channel IR means the network's very first layer is not pre-trained in any meaningful sense. The entire backbone's feature chain starts from a sub-optimal initialization.
3. **Progressive unfreezing train-val divergence is the Phase 2 ceiling**: The train-val loss gap grew by 1731% during Phase 2 (from 0.00005 to 0.00101). This means fine-tuning the full backbone created overfitting pressure rather than domain adaptation. The 80-subject dataset (~1700 training images) is insufficient to fully adapt 915 backbone parameter groups.
4. **Scratch models have an implicit advantage via initialization scale**: When training from scratch, all 1823 parameter tensors are Xavier/He initialized to be appropriate for small-scale spatial data (thermal IR). The pre-trained model starts all layers scaled for large, diverse ImageNet statistics — this requires the optimizer to do extra work to rescale distributions, consuming capacity that could otherwise improve localization.
5. **Sigma curriculum (Loop 17) explains the scratch-model advantage**: Loop 17's 43.1% PCK uses a sigma curriculum (3.0→1.5), which progressively trains the model to make finer-grained predictions. Pretrained models (fixed sigma=2.0) have no equivalent mechanism to force progressive localization refinement.

**FINAL VERDICT**: The pretrained HRNet-W32 approach is definitively abandoned as a primary strategy for this dataset. The ceiling is structurally imposed by domain gap, conv1 limitation, and insufficient data scale for backbone adaptation. All future loops will focus on scratch-based improvements.

## Next Planned Steps
1. **Loop 26 — Sigma Curriculum + Extended Robust Augmentations (Scratch, Subjects 1-80)**: Combine the 5-component stack (Sigma Curriculum, Structured Cutout, Thermal Intensity Jitter, Sensor Noise, Random Translation) to directly target the cover-type domain gap. Run for 40 epochs on full 80-subject set on the `feat/curriculum-robust-aug` branch. Expected peak: 47-52% PCK@0.2.
2. **Loop 27 (Contingent)**: If Loop 26 beats 46.6%, stack the hinge loss from Loop 9 (foreshortening prior) on top of the Loop 26 recipe.
