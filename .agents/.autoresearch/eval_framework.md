# Eval Framework

## Primary Metric Definitions
- **PCK@0.5**: Percentage of Correct Keypoints (Correct if distance < 0.5 * torso_diameter).
- **MPJPE**: Mean Per Joint Position Error (in pixels).

## Execution Commands
- Remote training: `python scripts/remote_train.py --run_id [ID] --eval` (Automatically triggers evaluation)
- Remote evaluation (Manual): `python scripts/remote_evaluate.py --run_id [ID] --checkpoint_name epoch_30.pth`

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
| Loop 10: Angle Constraints | 67.0% | 33.4 | 2026-05-10 |
| Loop 11: Joint Weighting | 73.1% | 28.0 | 2026-05-10 |
| Loop 12: Consistency | 75.6% | 26.2 | 2026-05-10 |
| Loop 13: Multi-Scale | 74.3% | 26.7 | 2026-05-10 |
| Loop 14: Integral Reg | 78.5% | 22.6 | 2026-05-10 |
| Loop 15: Occl-Aware | 81.0% | 20.7 | 2026-05-10 |
| Loop 16: Sigma Curric | 84.6% | 18.5 | 2026-05-10 |
