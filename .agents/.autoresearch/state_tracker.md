# State Tracker

- **Current Loop**: 25
- **Phase**: Planning (Phase 2)
- **Status**: Loop 24 completed. Pre-training approach still underperforms scratch baseline by 9.9% absolute. Root cause identified (see Diagnostic below). Strategy pivot being planned.
- **Absolute Priority**: 
  1. **Close the Pretrained Gap OR Pivot**: The pre-training gap is now well-understood. Next loop will test the highest-ROI hypothesis to either finally close it or confirm a pivot to other directions.
  2. **Scratch Baselines remain superior**: loop2 (46.6%), loop9 (45.1%), loop17 (43.1%) are all still ahead.
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

## ⚠️ CRITICAL: Pre-training Diagnostic (2026-05-18)

**Root causes identified for pretrained weight underperformance vs. scratch:**

1. **Epoch 1 PCK gap is the smoking gun**: Loop 23 (frozen backbone) starts at 9.81% PCK at epoch 1; Loop 24 (unfrozen, discriminative LR 0.1×) starts at 3.03%. The backbone features are already disrupted in the *very first epoch* even at backbone LR = 1e-5. Selective freezing is more protective than discriminative LR during head warmup.
2. **Training data scope mismatch**: Loop 23 used `subjects_train: [1, 80]` (80 subjects); Loop 24 used `[1, 30]` (30 subjects) for a fair comparison with loop2 scratch baseline. This ~2.67× data reduction compounds the convergence problem.
3. **Loop 23 PCK is noisy/unstable** (±5% variance epoch-to-epoch after ep 14): The frozen backbone creates a rigid feature extractor that eventually causes the model to overfit to its own feature representations. Val PCK plateaus and oscillates. This means the 41.0% best checkpoint is an outlier epoch, not a stable plateau.
4. **Pretrained model has a data efficiency advantage at the START but a domain ceiling**: At epochs 1-14, Loop 23 (pretrained+frozen) is 10-13% ahead. But after epoch 17, Loop 24 (unfrozen) begins catching up because it can adapt its backbone. If Loop 24 ran 60+ epochs, it would likely overtake Loop 23.
5. **The pretrained approach IS viable** but needs a 2-phase strategy: (Phase 1) warm up head with backbone frozen for ~10-15 epochs, THEN (Phase 2) unfreeze backbone with discriminative LR. This is the standard recipe in modern transfer learning (progressive unfreezing, ULMFiT-style).

## Next Planned Steps
1. **Loop 25 — Two-Phase Progressive Unfreezing**: Start with frozen backbone (as in Loop 23) for 10 epochs, then automatically unfreeze and apply discriminative LR for the remaining 20-40 epochs. This combines the early stability of Loop 23 with the late adaptability of Loop 24.
2. **Alternatively — Abandon Pretrained Route**: If Loop 25 still underperforms, abandon the pretrained approach entirely and focus on scratch-based improvements (Cutout augmentation, Visibility-Aware Attention, sigma curriculum tuning).
