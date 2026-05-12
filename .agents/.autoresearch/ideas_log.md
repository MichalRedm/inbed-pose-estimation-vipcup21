# Ideas Log

## Hypothesis Queue

> ⚠️ **Hypothesis #1 is an ABSOLUTE PREREQUISITE** — All previous results are suspect due to bugs in the evaluation script and loss landscape issues. No new experimental work should be done until the baseline is solidly re-established.

1. **[SUCCESS] Systematic Re-evaluation & Loss Alignment**: 
   - **Status**: Implementation complete (2026-05-11). `UncertaintyWeighting` added to `StandardTrainer`. `best_model.pth` now saved based on `val_pck`.
   - **Next**: Verify in Loop 17 training run.

2. **Feature-Level Multimodal Fusion (IR + PM)**:
   - **Hypothesis**: Pressure Maps (PM) provide absolute contact priors that are invariant to blanket thickness, while IR provides high-resolution texture. Fusing them will improve localization of occluded joints (ankles, knees) under thick blankets.
   - **Implementation**: Modify `VIPCupDataset` to load PM; update `HRNet` to support multi-channel input (2 channels: IR + PM); retrain.
   - **Status**: [BLOCKED] PM data not found in local dataset.

3. **Spatial Dependency Refinement (GCN)**:
   - **Hypothesis**: Joints are anatomically constrained. A GCN refinement layer taking soft-argmax coordinates can correct "impossible" poses (e.g., disconnected limbs) that occur under heavy occlusion.
   - **Result**: [FAILURE] (Loop 18). GCN layer caused excessive smoothing/regularization, dropping PCK to ~42%.

4. **Joint-Specific Adaptive Loss Scaling (Focal Heatmap Loss)**:
   - **Hypothesis**: Hard joints (ankles, wrists) are neglected during training as the model minimizes global MSE on easier joints (head, torso). Dynamic weighting based on per-joint error will force convergence on extremities.
   - **Implementation**: Track per-joint PCK during training; scale heatmap MSE weights inversely to PCK.

## Web Research Syntheses
- **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
- **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
- **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.

## Graveyard
- **Adversarial UDA (Global)**: Loop 5. Caused feature washout.
- **Fixed-Length MSE Anatomical Loss**: Loops 6-8. Over-regularized foreshortened poses, leading to -3.0% PCK regression. **PURGED** (2026-05-12).
- **Anatomical Angle Constraints (2D Hinge)**: Loop 10. Penalized sharp 2D projections of valid 3D poses, causing -9.4% PCK regression. 2D inner angle is not a reliable proxy for 3D ROM.
- **Joint-Specific Loss Weighting**: Loop 11. Over-focus on extremities (2.0x weight) led to structural instability and a -3.3% PCK regression compared to the baseline.
- **Occlusion Consistency Regularization**: Loop 12. Enforcing $pred_{occluded} \approx pred_{clean}$ improved hip stability but didn't beat the baseline on extremities.
- **Multi-Scale Heatmap Supervision**: Loop 13. Supervising intermediate resolutions (1/8, 1/16) caused gradient noise and a -2.1% PCK regression.
- **Normalized Skeleton Collapse**: Loop 19. Using anatomical hinge loss on 0-1 normalized coordinates without strong heatmap anchoring caused all joints to collapse into a single point to minimize bone length error (loss reached 0.00009).
- **Thermal Diffusion (Initial)**: Sign error in rotation augmentations.

## Current Iteration
- **Loop 6/8**: Fixed-Length MSE & Curriculum (FAILURE). **PURGED** due to regression and corrupted checkpoints.
- **Loop 9**: Foreshortening-Aware Hinge Loss (SUCCESS). Reached 76.4% PCK.
- **Loop 10**: Anatomical Angle Constraints (FAILURE). Regression to 67.0% PCK.
- **Loop 11**: Joint-Specific Loss Weighting (FAILURE). Regression to 73.1% PCK.
- **Loop 12**: Consistency Regularization (FAILURE). 75.6% PCK.
- **Loop 13**: Multi-Scale Heatmap Supervision (FAILURE). 74.3% PCK.
- **Loop 14**: Differentiable Heatmap Refinement (FAILURE - Audit). Corrected: 30.5% PCK.
- **Loop 15**: Occlusion-Aware Integral Regression (FAILURE - Audit). Corrected: 31.9% PCK.
- **Loop 16**: Adaptive Gaussian Sigma Curriculum (FAILURE - Audit). Corrected: 33.9% PCK.
- **Loop 17**: Multi-Task Uncertainty Weighting (SUCCESS). 74.2% PCK.
- **Loop 18**: GCN Refinement (FAILURE). 42.1% PCK.
- **Loop 19**: Normalized Anatomical Hinge (FAILURE). 39.0% PCK. Skeleton collapse.
