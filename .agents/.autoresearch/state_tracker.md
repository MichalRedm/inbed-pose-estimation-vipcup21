# State Tracker

- **Current Loop**: 36 (implementation)
- **Phase**: Phase 3 — Implementation
- **Status**: Loop 35 (**`loop35_jssca_attention`**) has concluded. It explored Joint-Symmetric Spatial-Channel Attention (JSSCA) to coordinate predicted keypoint heatmaps using an HRNet-W32 backbone with image replication. It achieved a peak validation of **66.56% PCK@0.2** and **17.60 px MPJPE**, beating the previous Loop 31 champion by +2.56pp and establishing a new SOTA record. However, spatial collapse inside the attention downsampling block was identified as a major bottleneck. We are proceeding to Loop 36 (JSSCA-v2 Option A: Backbone-Aware Joint-Spatial Neck Attention) to achieve >70% PCK.
- **Absolute Priority**:
  1. **Record**: Loop 35 (**66.56% PCK@0.2**, **17.60 px MPJPE**) is the new all-time record.
  2. **Next Step**: Design and implement Loop 36 (JSSCA-v2 Option A).
- **Baseline**: Loop 35 (66.56% PCK@0.2).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.2 (strict) | MPJPE | Status |
|-----|---------|--------------------|--------------------|--------|
| **loop35_jssca_attention** | argmax | **66.56%** | **17.60 px** | **NEW ALL-TIME RECORD** |
| loop31_improved_cover | argmax | **64.0%** | **17.79 px** | PREVIOUS RECORD champion |
| loop29_channel_replication | argmax | **52.0%** | 29.3 px | PREVIOUS TOP PCK |
| loop27_clean_sigma_cutout | argmax | **50.3%** | 27.2 px | SUCCESS |
| loop2_fixed_aug | argmax | **46.6%** | 29.6 px | Solid Scratch Baseline |
| loop9_anatomical_hinge | soft-argmax | **45.1%** | 25.3 px | RELIABLE |
| loop3_improved_thermal | argmax | 44.7% | 27.4 px | SUCCESS |
| loop7_anatomical_v2 | argmax | 44.1% | 36.0 px | STABLE |
| loop17_uncertainty | soft-argmax | **43.1%** | 24.5 px | HIGH-ACCURACY |
| loop5_uda_refined | argmax | 36.2% | 31.4 px | UDA REF |
| loop4_uda_alignment | argmax | 35.6% | 40.3 px | UDA BASE |
| loop23_stabilized_pretraining | argmax | **41.0%** | 37.0 px | FINE-TUNED |
| loop18_gcn_final_v5 | soft-argmax | 33.4% | 26.6 px | SUCCESS |
| loop34_kinematic_refinement | soft-argmax + Kinematic | 33.1% | 24.9 px | FAILURE |
| loop26_sigma_cutout | soft-argmax | 44.4% | 30.3 px | COLLAPSED |
| loop19 | soft-argmax | 12.7% | 66.7 px | COLLAPSED |

The loop16 `best_model.pth` was saved based on **combined val loss** (heatmap MSE + coord L1 + anatomical), NOT on PCK. Combined loss is dominated by the anatomical term (lambda=0.5) and does not align with PCK. The actual best PCK epoch for loop16 is unknown because only epoch_1.pth (corrupted) and best_model.pth were downloaded.

## ⚠️ CRITICAL: Prerequisite Issue Resolved

The fundamental loss-metric alignment problem has been resolved:
- The `best_model.pth` saving criterion is now tied directly to **val PCK`** (implemented in `base_trainer.py`).
- Atomic checkpoint downloads and integrity checks are now fully functional, protecting the local and remote directories against corruption during active runs.
- Uvicorn reload scope has been limited strictly to the `src/` directory, preventing accidental training stops during manual code/log audits.

## Infrastructure Fixes Applied (2026-05-12)

- **Model Self-Containment**: Standardized all legacy checkpoints to include weights, config, and decoding settings in a single `.pth` file.
- **Decoding Autopilot**: Implemented `PoseDecodingWrapper` to automatically apply the best decoding method (argmax vs soft-argmax) at inference time based on embedded metadata.
- **API Stabilization**: Resolved `NameError` and type-hinting issues in `src/api/main.py`. Ensured robust loading of historical checkpoints.
- **Corruption Recovery**: Restored `loop19` and `loop20` by recovering weights from `latest_model.pth` after `best_model.pth` was corrupted during save.
- **Remapping Fix**: Resolved a regression in `load_model_for_inference` where structural remapping (modules_list removal) failed for prefixed keys (e.g., `hrnet.`), which previously broke GCN-refined models like `loop18`.
- **Environment**: Enforced `.venv\Scripts\python.exe` for all backend services.
- **API & Remote Training Resiliency (2026-05-21)**: Restricted Uvicorn reload watcher scope strictly to the `src/` folder to prevent active runs from being interrupted by metadata or log writes. Implemented atomic checkpoint synchronization (writing to `.tmp` first, verifying PyTorch header/sizes, and renaming) to eliminate corrupt checkpoint risks completely.
- **Automatic Connection Recovery (2026-05-22)**: Implemented robust SSH keep-alive check and auto-retry harness within `GPUSession` and `TrainingManager`. The pipeline now seamlessly survives transient Wi-Fi drops, Cloudflare tunnel resets, and WinError socket drop codes without terminating active runs.

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
| 26 | Sigma Curriculum + Cutout (Scratch, 40ep) | FAILURE (BUGS) | **44.4%** | **PIPELINE BUGS**: (1) Horizontal flip keypoints were not re-indexed (coordinates scrambled under 50% flip); (2) Dynamic sigma curriculum failed to sync to CPU dataloader worker processes (sigma stayed at 3.0). Resulted in joint coalescence and coordinate collapse. Bugs are now fully fixed and verified locally. |
| 27 | Clean Rerun of Sigma Curriculum + Cutout (Scratch, 40ep) | SUCCESS | **50.3%** | Resolved worker curriculum desync via dynamic GPU-based target generation, and corrected keypoint swap indexing for horizontal flips. Reached a groundbreaking PCK of **50.3%** (beating scratch baseline by **+3.7pp**) and **27.2 px MPJPE** with zero skeleton collapse. |
| 28 | Stacking Two-Sided Hinge + Local Soft-Argmax | FAILURE | 43.9% (Argmax) / 41.9% (Soft-Argmax) | **FAULTY APPROACH**: Severely failed on simple uncovered examples (crossed ankles, wrist double-predictions). Reverted completely; re-established Loop 27. |
| 29 | Input Channel Replication (3-channel IR) | SUCCESS | **52.0%** | **NEW TOP PCK**. Hypothesis confirmed: preserving ImageNet conv1 priors via replication bridges the domain gap. |
| 30 | Discriminative LR (0.1x backbone) | FAILURE | 31.6% | **BACKBONE UNDERFITTING**: 1e-5 backbone LR was too low to adapt to IR features. |
| 31 | Physically-Realistic fabric draping + Structured Cutout + Sigma Curriculum + Channel Replication (40ep) | SUCCESS | **64.0%** | **ALL-TIME RECORD**. Hypothesis verified: high-fidelity blanket drape and wrinkle simulation + structured cutout forces the network to learn holistic geometric structures, while ImageNet Edge/Shape priors and curriculum learning provide a perfect adaptation path. |
| 32 | Cross-Modality Feature Distillation (MSE on Stage 3/4) | PARTIAL | ~55.0% | **NEGATIVE TRANSFER**: The teacher's raw RGB features have different texture statistics than IR. The student learned to map wrong textures. |
| 33 | Improved Output-Heatmap Distillation (KL Div) + Phase 2 Bypass | FAILURE | **56.2%** | **TEACHER CONFLICT**: Despite output-only distillation and linear decay, the RGB teacher's supervision on clean synthetic-blanket images actively contradicted the physical fabric drape augmentations, hurting the student's ability to learn thermal-specific occlusion physics. Peak PCK was 58.0% (Epoch 40), but local strict PCK is 56.2%. |
| 34 | Kinematic Bone-Vector Decomposition | FAILURE | **33.1%** | **CUMULATIVE ERROR PROPAGATION**: Decoupling keypoint prediction into root position + bone directions and lengths recursively reconstructed via forward kinematics accumulates directional and length errors recursively. Extremities (ankles/wrists) reached extremely low PCKs (8-16%) due to error stacking over 3-4 steps, and soft-argmax input caused severe spatial smoothing. Heatmap-based peak detection remains vastly superior. |
| 35 | Joint-Symmetric Spatial-Channel Attention (JSSCA) | SUCCESS | **66.56%** | **NEW ALL-TIME RECORD**. Hypothesis verified: spatial multi-head joint self-attention models limb dependencies and corrects extremity failures. However, average pooling the spatial features of heatmaps to $1\times 1$ tokens is highly lossy and limits localization accuracy. |

## Next Planned Steps (Post-Loop 35)

1. **Design and Implement JSSCA-v2 (Option A)**: Embed JSSCA *before* the output head, allowing it to act as a deep feature-refinement neck processing the multi-resolution features `(B, 480, 64, 64)` of HRNet-W32. Flatten joint-spatial dimensions to sequence tokens to perform spatial-joint co-attention directly in high-resolution coordinate space.
2. **Train & Evaluate**: Run Loop 36 for 40 epochs on remote Kaggle dual T4 GPUs. Compare metrics against JSSCA-v1 (66.56%) to break through the 70% threshold.
