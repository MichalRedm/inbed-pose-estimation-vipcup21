# State Tracker

- **Current Loop**: Optimization Complete
- **Phase**: Evaluation & Reporting
- **Status**: Pipeline stabilized at 84.6% PCK. 8.2% improvement over baseline.
- **Baseline**: Loop 16 (84.6% PCK, 18.5px MPJPE)

### Loop 8: Anatomical Curriculum Warmup - FAILURE
- [x] Implement linear warmup for `lambda_anatomical` in `StandardTrainer`.
- [x] Execute `loop8_anatomical_curriculum` (λ=0.5 target, 10-epoch warmup).
- [x] **RESULT: FAILURE (PCK@0.5: 73.1%)**.
- [x] **ROOT CAUSE**: Improved over Loop 7 (+1.2%) but still below baseline (-1.8%). MSE bone length prior is too rigid and penalizes natural 2D foreshortening, particularly for the L_Knee.

### Loop 9: Foreshortening-Aware Hinge Loss - SUCCESS
- [x] Implement `AnatomicalHingeLoss` (penalize only length > target).
- [x] Execute `loop9_anatomical_hinge` (λ=0.5 target, 10-epoch warmup).
- [x] **RESULT: SUCCESS (PCK@0.5: 76.4%)**.
- [x] **ACTION**: Surpassed baseline (74.9%). Hinge loss allows for 2D foreshortening while maintaining structural integrity.

| Loop ID | Baseline Metrics Table | Iteration Log (Loop ID, Hypothesis, Result, Action) |
|---------|------------------------|---------------------------------------------------|
| 3       | PCK@0.5: 74.9%, MPJPE: 27.4 | 3, loop3_improved_thermal_full_data, SUCCESS, Improved Thermal Diffusion (wavy edges, full cover) + Full Dataset (80 subjects). |
| 8       | PCK@0.5: 73.1%, MPJPE: 35.3 | 8, loop8_anatomical_curriculum, FAILURE, Curriculum stabilized but fixed 2D lengths hurt foreshortened poses. |
| 9       | PCK@0.5: 76.4%, MPJPE: 25.8 | 9, loop9_anatomical_hinge, SUCCESS, Using hinge loss to allow for natural limb foreshortening. |
| 10      | PCK@0.5: 67.0%, MPJPE: 33.4 | 10, loop10_angle_constraints, FAILURE, 2D angle hinge loss over-regularized foreshortened poses. |
| 11      | PCK@0.5: 73.1%, MPJPE: 28.0 | 11, loop11_joint_weighting, FAILURE, Joint weighting focused on extremities but destabilized core pose. |
| 12      | PCK@0.5: 75.6%, MPJPE: 26.2 | 12, loop12_consistency, FAILURE, Consistency helped hips but not ankles/wrists. |
| 13      | PCK@0.5: 74.3%, MPJPE: 26.7 | 13, loop13_multi_scale, FAILURE, Multi-scale supervision added gradient noise. |
| 14      | PCK@0.5: 78.5%, MPJPE: 22.6 | 14, loop14_integral_regression, SUCCESS, Soft-argmax coordinate regression improved precision. |
| 15      | PCK@0.5: 81.0%, MPJPE: 20.7 | 15, loop15_occlusion_aware_integral, SUCCESS, Supervising occluded joints stabilized extremities. |
| 16      | PCK@0.5: 84.6%, MPJPE: 18.5 | 16, loop16_sigma_curriculum, SUCCESS, Decaying sigma pinpointed joints. |
