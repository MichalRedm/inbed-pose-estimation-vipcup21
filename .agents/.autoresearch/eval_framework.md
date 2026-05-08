# Eval Framework

## Primary Metric Definitions
- **PCK@0.5**: Percentage of Correct Keypoints.
- **MPJPE**: Mean Per Joint Position Error.

## Execution Commands
- Local training: `python scripts/train.py`
- Remote training: `python scripts/remote_train.py`
- Baseline evaluation: `python scripts/evaluate.py` (Assuming this exists or will be created)

## Advanced Diagnostics
- Slice-based analysis: Compare performance on different subjects.
- Visual inspection: Dashboard visualization of predictions on covered subjects.

## Results Tracker
| Experiment | PCK@0.5 | MPJPE | Date |
|------------|---------|-------|------|
| Baseline (Poly Occlusion) | 42.9% | 65.5 | 2026-05-08 |
| Loop 1: Thermal Diffusion | 4.6% (Bug) | 106.2 | 2026-05-08 |
| Loop 2: fixed_aug (Stable) | 73.0% | 29.7 | 2026-05-08 |
