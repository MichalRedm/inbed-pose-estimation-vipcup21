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
Fresh evaluation on a run (use this instead of `scripts/evaluate.py`):
```python
# In scratch/eval_<run_id>.py — see base_trainer.compute_val_pck() for reference impl.
from src.models import build_model
from src.data.dataset import VIPCupDataset, collate_skip_none
from src.utils.pose import decode_heatmaps
import torch, json, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

run_id = "loop16_sigma_curriculum"
run_dir = Path(f"results/runs/{run_id}")
state = torch.load(run_dir / "checkpoints/best_model.pth", map_location="cpu")
cfg = state["config"]  # always use run's own config, NOT load_config()
model = build_model(cfg); model.load_state_dict(state["model_state_dict"]); model.eval()
image_size = tuple(cfg["dataset"].get("image_size", [256, 256]))
s_val = cfg["dataset"].get("subjects_val", [81, 90])
decode_method = "soft-argmax" if cfg["training"].get("sigma_start", 2.0) != cfg["training"].get("sigma_end", 2.0) else "argmax"
ds = VIPCupDataset(root=cfg["dataset"]["root"], subjects=range(s_val[0], s_val[1]+1),
                   modalities=["IR"], covers=["cover1", "cover2"], split="val", image_size=image_size)
loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_skip_none)
# ... decode and compute PCK with vis <= 1, torso = R_Shoulder(8) - L_Hip(3)
```

> ⚠️ **Do NOT use `scripts/evaluate.py`** directly — it loads the global default config instead of the run's config, uses `vis==0` only (misses occluded joints), and hardcodes `soft-argmax` for all models. These bugs cause inflated/incorrect metrics.

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
