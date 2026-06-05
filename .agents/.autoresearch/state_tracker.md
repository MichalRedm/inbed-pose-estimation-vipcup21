# State Tracker

- **Current Loop**: 57 (Refined Self-Training)
- **Phase**: Phase 5 — Recursive Continuation & State Logging
- **Status**: FINISHED. Implemented Adaptive Relative Confidence Thresholding and Dynamic Unlabeled Loss Weighting. This iteration successfully beat the previous record, achieving **86.7% PCK@0.2** and **8.6 px MPJPE**.
- **Absolute Priority**:
  - **Record**: Loop 57 (**86.7% PCK@0.2**, **8.6 px MPJPE**) is the **new stable verified record**.
  - **Next Goal**: Implement Task-Consistent Domain Translation (Sem-GAN/CUT with Pose Loss) or MoE Modality Routing.
- **Baseline**: Loop 57 (86.7% PCK@0.2).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.2 (strict) | MPJPE | Status |
|-----|---------|--------------------|--------------------|--------|
| **loop57_refined_self_training** | argmax | **86.7%** | **8.6 px** | **ALL-TIME RECORD** |
| loop56_tuned_self_training | argmax | **86.2%** | **8.9 px** | Legacy Record |
| loop55_adaptive_curriculum | argmax | **82.8%** | **10.2 px** | Legacy Record |
| loop54_self_training_v3 | argmax | **82.5%** | **10.2 px** | Stable Baseline |
| loop53_advanced_cover | argmax | **78.7%** | **11.9 px** | CNN/ViT Champion |
| **loop44_vitpose_fixed** | argmax | **77.8%** | **12.3 px** | Legacy Record |

## Iteration Log

| Loop ID | Hypothesis | Result | Corrected PCK@0.2 | Action |
|---------|-----------|--------|--------------|--------|
| 44 | Stabilized ViTPose Fine-tuning (Fixed Structure + Disc. LR) | SUCCESS | **77.8%** | Bypassing class token; backbone LR at 5e-6; stable sigma=3.0. **STABLE RECORD**. |
| 53 | Advanced Cover (FDA + HistMatch + Bank) | SUCCESS | **78.7%** | Refactored augmentations; dynamic reference bank improved realism. **NEW STABLE RECORD**. |
| 54 | Self-Training (EMA Teacher + CUT Strong Aug) | SUCCESS | **82.5%** | **v3 Run (Final Record)**: Achieved 82.5% PCK in a fresh, uninterrupted run. Verified the stability of EMA-based consistency regularization. |
| 55 | Adaptive Confidence Curriculum (0.6 -> 0.25) | SUCCESS | **82.8%** | Decaying threshold increased target utilization. Resumption bug fix ensured curriculum stability. **NEW STABLE RECORD**. |
| 56 | Tuned Self-Training (60 Epochs + Cosine Decay + Cosine EMA + Joint Discounts + Soft Weighting) | SUCCESS | **86.2%** | Cosine schedules, part-aware thresholds (extremities discounted), soft weighting, and 60-epoch budget stabilized training and accelerated adaptation. **NEW STABLE RECORD**. |
| 57 | Refined Self-Training (Adaptive relative thresholds + dynamic unlabeled loss weight λu) | SUCCESS | **86.7%** | Relative threshold ties joint thresholds to teacher conf EMA. Dynamic λu scales unlabeled loss based on teacher confidence. **NEW STABLE RECORD**. |

## Next Planned Steps

1. **Semantic & Pose-Consistent Domain Translation**: Replacing cycle consistency with pose-consistency loss to synthesize realistic blankets while anchoring pose geometry.
2. **MoE Modality Routing**: Route tokens between "Clean" and "Occluded" experts to handle capacity interference between source and target domains.
3. **Multi-View Consistency**: Leverage side views for cross-view pseudo-label regularization.
