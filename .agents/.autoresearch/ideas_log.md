# Ideas Log

## Hypothesis Queue
1. **Multimodal Fusion (Uncovered RGB/Pressure)**: Fuse features from available modalities to disambiguate occluded joints.
2. **Joint-Specific Loss Weighting**: Increase penalty for ankle/knee prediction errors.
3. **Adaptive Lambda**: Scale λ_ana based on the ratio of loss_pose / loss_ana.
4. **Consistency Regularization**: Force predictions to be invariant to occlusion-style augmentations.

## Web Research Syntheses
- **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
- **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
- **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.

## Graveyard
- **Adversarial UDA (Global)**: Loop 5. Caused feature washout.
- **Fixed-Length MSE Anatomical Loss**: Loops 6-8. Over-regularized foreshortened poses, leading to -3.0% PCK regression.
- **Thermal Diffusion (Initial)**: Sign error in rotation augmentations.

## Current Iteration
- **Loop 9**: Foreshortening-Aware Hinge Loss (SUCCESS). Reached 76.4% PCK.
