# State Tracker

- **Current Loop**: 35 (planning)
- **Phase**: Phase 1 — Deep Codebase & Online Research
- **Status**: Loop 34 (**`loop34_kinematic_refinement`**) has concluded. It explored differentiable Kinematic Bone-Vector Decomposition (decoupling keypoint prediction into root position + bone directions and lengths recursively reconstructed via forward kinematics) combined with Loop 31 augmentations. Despite correcting a modality-dependent bone scaling bug, the model reached only **33.1% PCK@0.2** and **24.9 px MPJPE** on the validation set. This represents a severe regression (-30.9pp) compared to the Loop 31 champion baseline of 64.0%.
- **Absolute Priority**:
  1. **Record**: Loop 31 (**64.0% PCK@0.2**, **17.79 px MPJPE**) remains the all-time record.
  2. **Next Step**: Design Loop 35. We will explore YOLO-Pose thermal pretraining (OpenThermalPose) or other competitive heatmap-based refinement.
- **Baseline**: Loop 31 (64.0% PCK@0.2).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.2 (strict) | MPJPE | Status |
|-----|---------|--------------------|--------------------|--------|
| **loop31_improved_cover** | argmax | **64.0%** | **17.79 px** | **NEW ALL-TIME RECORD** |
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
- The `best_model.pth` saving criterion is now tied directly to **val PCK** (implemented in `base_trainer.py`).
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

## ⚠️ CRITICAL: Pretrained Route Post-Mortem (2026-05-18 — Final)

**Root causes for the confirmed structural ceiling at ~42% PCK:**

1. **RGB→IR Domain Gap is fundamental, not addressable by scheduling**: ImageNet RGB features (texture, color, spatial gradients) have minimal overlap with thermal IR (heat diffusion, emissivity, body-mass heat signatures). The backbone, despite 50 epochs of fine-tuning, retains structural biases that are misaligned with the thermal domain. This is consistent with the literature: domain-specific pre-training (e.g., grayscale-COCO or thermal-dataset pre-training) is required to bridge this gap reliably.
2. **1-channel conv1 adaptation is a known weak link**: HRNet-W32 conv1 is designed for 3-channel RGB. Averaging the 3 input channels to adapt to 1-channel IR means the network's very first layer is not pre-trained in any meaningful sense. The entire backbone's feature chain starts from a sub-optimal initialization.
3. **Progressive unfreezing train-val divergence is the Phase 2 ceiling**: The train-val loss gap grew by 1731% during Phase 2 (from 0.00005 to 0.00101). This means fine-tuning the full backbone created overfitting pressure rather than domain adaptation. The 80-subject dataset (~1700 training images) is insufficient to fully adapt 915 backbone parameter groups.
4. **Scratch models have an implicit advantage via initialization scale**: When training from scratch, all 1823 parameter tensors are Xavier/He initialized to be appropriate for small-scale spatial data (thermal IR). The pre-trained model starts all layers scaled for large, diverse ImageNet statistics — this requires the optimizer to do extra work to rescale distributions, consuming capacity that could otherwise improve localization.
5. **Sigma curriculum (Loop 17) explains the scratch-model advantage**: Loop 17's 43.1% PCK uses a sigma curriculum (3.0→1.5), which progressively trains the model to make finer-grained predictions. Pretrained models (fixed sigma=2.0) have no equivalent mechanism to force progressive localization refinement.

**FINAL VERDICT**: The pretrained HRNet-W32 approach is definitively abandoned as a primary strategy for this dataset. The ceiling is structurally imposed by domain gap, conv1 limitation, and insufficient data scale for backbone adaptation. All future loops will focus on scratch-based improvements.

## Next Planned Steps (Post-Loop 34)
1. **Abandon Kinematic Refinement**: Cumulative error propagation along the recursively reconstructed tree structure underperforms direct heatmap keypoint regression.
2. **Design Loop 35**: Design and fine-tune a YOLO-pose based thermal keypoint estimator or explore alternative spatial self-attention/transformer-based keypoint tracking.e-Vector Decomposition.


