# Eval Framework

## Primary Metric Definitions
- **PCK@0.2**: Percentage of Correct Keypoints. A joint is correct if its Euclidean distance to GT is < **0.2** × torso_diameter, where torso_diameter = ‖R_Shoulder(idx 8) − L_Hip(idx 3)‖. This is the **strict academic standard** for high-precision pose estimation.
- **MPJPE**: Mean Per Joint Position Error in pixels. Evaluated on the same visibility mask.

## Evaluation Protocol

### Correct Validation Setup
- **Subjects**: val subjects 81–90 (10 subjects, defined in run's own `config.json`)
- **Covers**: `cover1` and `cover2` ONLY (covered images — this is the task target domain)
- **Image size**: from run's own `config.json → dataset.image_size` (default 256×256)
- **Decoder**: auto-selected per run:
  - `GCN Refinement` if model is `refined_hrnet` (evaluates `refined_coords` directly).
  - `soft-argmax` if `training.sigma_start != training.sigma_end` (sigma curriculum).
  - `argmax` otherwise (standard heatmap MSE).

### Running Evaluation
- [x] **Evaluation Script Fixed**: `scripts/evaluate.py` now loads run-specific configs, auto-selects decoders, and uses the correct `vis<=1` mask.

# Evaluate a specific run (auto-saves results to the run directory)
# IMPORTANT: Must be run from project root with absolute path resolution for visual audit plots.
python scripts/evaluate.py --run_id loop17_uncertainty
```

### Training-integrated Evaluation
From next run onward, `val_pck` is logged each epoch in `history.json` and `best_model.pth` is saved at the epoch of highest `val_pck`. See `src/training/base_trainer.py → compute_val_pck()`.

### Verification Protocol (Mandatory for Baselines)
All "Verified" metrics in this repository must meet the following criteria:
1. **Local Execution**: Evaluation MUST be run locally or via the fixed `scripts/evaluate.py` to ensure local config parity.
2. **Run-Specific Config**: Must use the `config.json` found in the run's own directory, NOT the global `default.yaml`.
3. **Correct Mask**: Must use `vis <= 1` (visible + occluded) for PCK/MPJPE.
4. **Decoder Match**: Must manually or automatically select the decoder (argmax vs soft-argmax) that matches the training method.
5. **Traceability**: The `history.json` and `best_model.pth` must be present and uncorrupted.

## Results Tracker (STRICT PCK@0.2 — Corrected Baselines)

| Run | PCK@0.2 | MPJPE | Status |
|-----|---------|-------|--------|
| **loop54_self_training_v3** | **82.5%** | **10.2 px** | **NEW ALL-TIME RECORD (Self-Training)** |
| **loop53_vitpose_advanced_cover** | **78.7%** | **11.9 px** | CNN/ViT Hybrid Champion |
| **loop50_vitpose_cut_aug** | **78.4%** | **11.8 px** | Marginal Peak (Overhead Check) |
| **loop44_vitpose_fixed** | **77.8%** | **12.3 px** | Solid Stable Baseline |
| loop52_vitpose_balanced | 77.6% | 12.1 px | Balanced Augmentations |
| loop51_vitpose_cut_boost | 76.6% | 12.5 px | Over-augmented (CUT Noise) |
| **loop31_improved_cover** | **64.0%** | **17.8 px** | PREVIOUS RECORD CHAMPION (CNN) |
| **loop29_channel_replication** | **52.0%** | 29.3 px | TOP BASELINE |
| **loop27_clean_sigma_cutout** | **50.3%** | 27.2 px | **TOP ROBUSTNESS** |
| **loop2_fixed_aug** | **46.6%** | 29.6 px | Solid Scratch Baseline |
| **loop9_anatomical** | **45.1%** | 25.3 px | RELIABLE |
| **loop17_uncertainty** | **43.1%** | **24.5 px** | HIGH-ACCURACY |
| loop23_stabilized_pretraining | 41.0% | 37.0 px | FINE-TUNED |
| loop18_gcn_refinement| 33.4% | 26.6 px | GCN Smoothing Collapse |
| loop24_unfrozen_pretraining | 36.7% | 41.3 px | UNDERPERFORMED |
| loop19_normalized_ana | 12.7% | 66.7 px | FAILURE (**Skeleton Collapse**) |

*Note: Translation tasks (Loop 47 CycleGAN, Loop 48 CUT) are physical augmentations evaluated visually, not via PCK, until they are integrated into a downstream pose estimator pipeline.*

> Previous figures (76.4%, 78.5%, 81.0%, 84.6% etc.) were computed by the flawed remote evaluate.py and were based on the less strict **PCK@0.5** metric. They are directionally useful (comparing relative improvement) but not accurate absolute baselines for the current **PCK@0.2** standard.

## Advanced Diagnostics
- **Loss-metric alignment check**: After each run, compare `val_loss` trajectory in `history.json` against `val_pck`. If the epoch with minimum `val_loss` differs significantly from the epoch with max `val_pck`, the loss function has an alignment problem. This is a known issue with the combined auxiliary loss.
- **Per-joint PCK breakdown**: Extremities (ankles, wrists) consistently underperform. Focus new hypotheses on improving R/L_Ankle PCK specifically.
- **Cover-specific breakdown**: Run evaluation separately on cover1 vs cover2 to detect if the model struggles more with thicker blanket conditions.
- **Visual Audit**: `scripts/evaluate.py` generates `visual_audit_best_model.png` in the run directory. This plot compares GT vs Pred skeletons across 4 validation samples. If the predicted skeleton is a single point, this confirms **Skeleton Collapse**.
- **Anatomical Validity Check**: For GCN-refined models, compare `PCK` of raw heatmaps vs `PCK` of refined coordinates to quantify the GCN's "correction" effect.
