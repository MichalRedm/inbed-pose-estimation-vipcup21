# Eval Framework

## Primary Metric Definitions
- **PCK@0.5**: Percentage of Correct Keypoints (Correct if distance < 0.5 * torso_diameter).
- **MPJPE**: Mean Per Joint Position Error (in pixels).

## Execution Commands
- Remote training: `python scripts/remote_train.py --run_id [ID] --lambda_anatomical [VAL]`
- Remote evaluation: `python scripts/remote_evaluate.py --run_id [ID] --checkpoint_name epoch_30.pth`

## Advanced Diagnostics
- **Anatomical Integrity**: Check `loss_ana` in history. Successful models should reach 0.0 with Hinge Loss.
- **Visual Inspection**: Use the dashboard to check L_Knee predictions on "cover2" subjects.

## Results Tracker
| Experiment | PCK@0.5 | MPJPE | Date |
|------------|---------|-------|------|
| Loop 3 (Baseline) | 74.9% | 27.4 | 2026-05-09 |
| Loop 7: Anatomical V1 | 71.9% | 36.1 | 2026-05-09 |
| Loop 8: Curriculum | 73.1% | 35.3 | 2026-05-10 |
| Loop 9: Hinge Loss | 76.4% | 25.8 | 2026-05-10 |
