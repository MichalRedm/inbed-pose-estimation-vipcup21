# Eval Framework

## Primary Metric Definitions
- **PCK@0.5**: Percentage of Correct Keypoints. A joint is correct if its Euclidean distance to GT is < 0.5 × torso_diameter, where torso_diameter = ‖R_Shoulder(idx 8) − L_Hip(idx 3)‖. Evaluated on **all joints with visibility ≤ 1** (visible + occluded under blanket).
- **MPJPE**: Mean Per Joint Position Error in pixels. Evaluated on the same visibility mask.

## Evaluation Protocol

### Correct Validation Setup
- **Subjects**: val subjects 81–90 (10 subjects, defined in run's own `config.json`)
- **Covers**: `cover1` and `cover2` ONLY (covered images — this is the task target domain)
- **Image size**: from run's own `config.json → dataset.image_size` (default 256×256)
- **Decoder**: auto-selected per run:
  - `soft-argmax` if `training.sigma_start != training.sigma_end` (sigma curriculum)
  - `argmax` otherwise (standard heatmap MSE)

### Running Evaluation
- [x] **Evaluation Script Fixed**: `scripts/evaluate.py` now loads run-specific configs, auto-selects decoders, and uses the correct `vis<=1` mask.

```bash
# Evaluate a specific run (auto-saves results to the run directory)
python scripts/evaluate.py --run_id loop16_sigma_curriculum
```

### Training-integrated Evaluation
From next run onward, `val_pck` is logged each epoch in `history.json` and `best_model.pth` is saved at the epoch of highest `val_pck`. See `src/training/base_trainer.py → compute_val_pck()`.

## Results Tracker (CORRECTED — cover1+cover2, vis≤1, run config, correct decoder)

| Experiment | PCK@0.5 | MPJPE | Notes |
|------------|---------|-------|-------|
| Loop 9: Hinge Loss | ~78% | ~27 px | vis==0 only; true vis≤1 not yet measured |
| Loop 16: Sigma Curriculum | **78.8%** | 26.4 px | Verified 2026-05-11 |

> Previous figures (76.4%, 78.5%, 81.0%, 84.6% etc.) were computed by the flawed remote evaluate.py. They are directionally useful (comparing relative improvement) but not accurate absolute baselines.

## Advanced Diagnostics
- **Loss-metric alignment check**: After each run, compare `val_loss` trajectory in `history.json` against `val_pck`. If the epoch with minimum `val_loss` differs significantly from the epoch with max `val_pck`, the loss function has an alignment problem. This is a known issue with the combined auxiliary loss.
- **Per-joint PCK breakdown**: Extremities (ankles, wrists) consistently underperform. Focus new hypotheses on improving R/L_Ankle PCK specifically.
- **Cover-specific breakdown**: Run evaluation separately on cover1 vs cover2 to detect if the model struggles more with thicker blanket conditions.
