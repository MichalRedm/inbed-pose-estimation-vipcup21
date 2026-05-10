# Ideas Log

## Hypothesis Queue
1. **Multimodal Fusion (IR + PM)**: Fuse features from IR (high-res) and Pressure Maps (absolute contact) to disambiguate joints under thick covers.
2. **Anatomical Angle Constraints**: Implement hinge losses for joint angles (e.g. elbows, knees) to prevent anatomically impossible poses.
3. **Joint-Specific Uncertainty Weighting**: Learn per-joint uncertainty to adaptively weigh the loss, focusing on unambiguous joints.
4. **Consistency Regularization**: Force predictions to be invariant to occlusion-style and spatial augmentations.

## Web Research Syntheses
- **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
- **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
- **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.

## Graveyard
- **Adversarial UDA (Global)**: Loop 5. Caused feature washout.
- **Fixed-Length MSE Anatomical Loss**: Loops 6-8. Over-regularized foreshortened poses, leading to -3.0% PCK regression.
- **Anatomical Angle Constraints (2D Hinge)**: Loop 10. Penalized sharp 2D projections of valid 3D poses, causing -9.4% PCK regression. 2D inner angle is not a reliable proxy for 3D ROM.
- **Joint-Specific Loss Weighting**: Loop 11. Over-focus on extremities (2.0x weight) led to structural instability and a -3.3% PCK regression compared to the baseline.
- **Occlusion Consistency Regularization**: Loop 12. Enforcing $pred_{occluded} \approx pred_{clean}$ improved hip stability but didn't beat the baseline on extremities.
- **Multi-Scale Heatmap Supervision**: Loop 13. Supervising intermediate resolutions (1/8, 1/16) caused gradient noise and a -2.1% PCK regression.
- **Thermal Diffusion (Initial)**: Sign error in rotation augmentations.

## Current Iteration
- **Loop 9**: Foreshortening-Aware Hinge Loss (SUCCESS). Reached 76.4% PCK.
- **Loop 10**: Anatomical Angle Constraints (FAILURE). Regression to 67.0% PCK.
- **Loop 11**: Joint-Specific Loss Weighting (FAILURE). Regression to 73.1% PCK.
- **Loop 12**: Consistency Regularization (FAILURE). 75.6% PCK.
- **Loop 13**: Multi-Scale Heatmap Supervision (FAILURE). 74.3% PCK.
- **Loop 14**: Differentiable Heatmap Refinement - PLANNED.
