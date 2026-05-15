# Ideas Log

## Hypothesis Queue

> ⚠️ **Hypothesis #1 is an ABSOLUTE PREREQUISITE** — All previous results are suspect due to bugs in the evaluation script and loss landscape issues. No new experimental work should be done until the baseline is solidly re-established.

1. **[SUCCESS] Systematic Re-evaluation & Loss Alignment**: 
   - **Status**: Implementation complete (2026-05-11). `UncertaintyWeighting` added to `StandardTrainer`. `best_model.pth` now saved based on `val_pck`.
   - **Next**: Verify in Loop 17 training run.

2. **[ACTIVE] Stabilized Transfer Learning via Selective Freezing (Loop 23)**:
   - **Hypothesis**: The underperformance of pre-trained HRNet in Loops 20/21 was caused by **feature washout** (gradients from the random head destroying backbone features) and **underfitting**. By **freezing the Stem and Stage 1** (generic edge/texture detectors) and using a **lower LR (5e-5)**, the model can quickly adapt the high-level pose logic to thermal data without losing the ImageNet structural priors. This should match the convergence speed of scratch models while surpassing the 46.6% PCK baseline.
   - **Implementation**: Set `requires_grad=False` for `conv1`, `conv2`, and `layer1`. Use Argmax decoding.
   - **Priority**: Absolute (Highest ROI - minimal change for likely high gain).

3. **Visibility-Aware Attention Masking (Auxiliary Visibility Branch)**:
   - **Hypothesis**: The model struggles with thick blanket occlusions because it processes visible and occluded joints identically. Adding an auxiliary branch to predict the `vis` flag (visible vs. occluded) and using its output to modulate spatial features (via an attention mask) will force the network to explicitly differentiate between reliable thermal signatures and areas where it must rely on structural priors.
   - **Implementation**: Add a small classification head to HRNet to predict visibility per joint. Use this prediction to scale or attend to the feature maps before heatmap generation.
   - **Priority**: High (Data is already annotated with `vis` flags, minimal architectural change, directly addresses occlusion).

4. **Structured Regional Cutout (Simulated Extreme Occlusion)**:
   - **Hypothesis**: The current `ThermalDiffusionAugmenter` only applies a wavy blur to simulate a blanket. The model relies too much on residual thermal leakage. Applying "GridMask" or large contiguous "Cutout" blocks that zero out entire limbs will force the network to learn holistic structural dependencies rather than local textures, preparing it for the extreme occlusion of `cover2`.
   - **Implementation**: Add a simple structured cutout augmentation that completely masks 25-50% of the image during training.
   - **Priority**: Medium-High (Very simple to implement in `DataAugmenter`, specifically targets the domain gap).

5. **Kinematic Bone-Vector Decomposition (Decoupled Length and Direction)**:
   - **Hypothesis**: Direct regression of absolute (x,y) coordinates under anatomical constraints often leads to "skeleton collapse" (Loop 19) because minimizing bone length to 0 perfectly satisfies the Hinge loss. Decoupling the prediction into a root joint (pelvis) plus bone vectors (length and angle) prevents this. The length can be strongly regularized while angles vary freely.
   - **Implementation**: Modify the `SoftArgmax2D` layer or the prediction head to regress root (x,y) and relative vectors for limbs, reconstructing the final pose via forward kinematics.
   - **Priority**: Medium (Strong theoretical guarantee against collapse, but requires non-trivial architectural and loss restructuring).

6. **Feature-Level Multimodal Fusion (IR + PM)**:
   - **Status**: [BLOCKED] Pressure Maps (PM) data not found in local dataset.

7. **Spatial Dependency Refinement (GCN)**:
   - **Result**: [FAILURE] (Loop 18). GCN layer caused excessive smoothing.

8. **Joint-Specific Adaptive Loss Scaling (Focal Heatmap Loss)**:
   - **Result**: [FAILURE] (Loop 11). Over-focus on extremities led to structural instability.

## Web Research Syntheses
- **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
- **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
- **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.
- **Loop 20 Synthesis (HRNet Underperformance)**: ImageNet-pre-trained HRNet-W32 (35.1% PCK) failed to beat the lighter baseline (46.6%). Analysis shows linear loss decrease at epoch 30, indicating significant underfitting. Additionally, the 1-channel `conv1` was randomly initialized, breaking the low-level feature extraction chain. Doubling epochs and using averaged `conv1` weights is required.
- **Preventing Skeleton Collapse**: Research indicates direct coordinate regression with structural penalties often leads to collapse. State-of-the-art methods decompose pose into root position + bone vectors (length/angle), applying length priors without compressing the skeleton.
- **Occlusion Handling**: Multi-modal fusion is best, but when restricted to IR, explicitly modeling visibility (e.g., through an auxiliary attention branch) helps the network switch from texture-reliance to prior-reliance.
- **Loop 22 Analysis**: Established that Soft-Argmax causes "Joint Coalescence" in thermal images due to the "center-of-mass" being pulled toward high-intensity heat (head). Returning to Argmax is essential for precision.

## Graveyard
- **Adversarial UDA (Global)**: Loop 5. Caused feature washout.
- **Fixed-Length MSE Anatomical Loss**: Loops 6-8. Over-regularized foreshortened poses, leading to -3.0% PCK regression. **PURGED** (2026-05-12).
- **Anatomical Angle Constraints (2D Hinge)**: Loop 10. Penalized sharp 2D projections of valid 3D poses, causing -9.4% PCK regression. 2D inner angle is not a reliable proxy for 3D ROM.
- **Joint-Specific Loss Weighting**: Loop 11. Over-focus on extremities (2.0x weight) led to structural instability and a -3.3% PCK regression compared to the baseline.
- **Occlusion Consistency Regularization**: Loop 12. Enforcing $pred_{occluded} \approx pred_{clean}$ improved hip stability but didn't beat the baseline on extremities.
- **Multi-Scale Heatmap Supervision**: Loop 13. Supervising intermediate resolutions (1/8, 1/16) caused gradient noise and a -2.1% PCK regression.
- **Normalized Skeleton Collapse**: Loop 19. Using anatomical hinge loss on 0-1 normalized coordinates without strong heatmap anchoring caused all joints to collapse into a single point to minimize bone length error (loss reached 0.00009).
- **Thermal Diffusion (Initial)**: Sign error in rotation augmentations.

- **Loop 21 Enhanced HRNet (Pre-trained + Soft-Argmax + 60 Epochs)**: Loop 21. Achieved only 32.0% PCK. **VISUAL FAILURE**: Inference on training examples shows severe **Joint Coalescence** in the head area. Multiple joints (extremities, torso) are clustered around the head, indicating that the Soft-Argmax 'center-of-mass' calculation is being pulled toward the highest intensity thermal region, and the Coordinate Regression loss is not strong enough to pull them away, or is actively encouraging this clustering to minimize global error. This confirms that Soft-Argmax is currently detrimental to precision.

## Current Iteration
- **Loop 9**: Foreshortening-Aware Hinge Loss (SUCCESS). Reached 45.1% PCK@0.2.
- **Loop 17**: Multi-Task Uncertainty Weighting (SUCCESS). 43.1% PCK@0.2.
- **Loop 18**: GCN Refinement (FAILURE). 33.4% PCK@0.2.
- **Loop 19**: Normalized Anatomical Hinge (FAILURE). 12.7% PCK@0.2. Skeleton collapse.
- **Loop 20/21**: Pre-trained HRNet + Soft-Argmax (FAILURE). 32.0% PCK@0.2. Coordinate regression is hindering precision.
