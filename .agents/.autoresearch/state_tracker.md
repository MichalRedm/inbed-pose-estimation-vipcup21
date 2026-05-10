# State Tracker

- **Current Loop**: Loop 10 (Planned: Multimodal Fusion)
- **Phase**: Phase 0 (Workspace Initialization)
- **Status**: Ready for Loop 10.
- **Baseline**: Loop 9 (76.4% PCK, 25.8px MPJPE)

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
