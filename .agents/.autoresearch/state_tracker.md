# State Tracker

- **Current Loop**: 55 (Adaptive Confidence Curriculum)
- **Phase**: Phase 4 — Evolving Evaluation Framework
- **Status**: FINISHED. Implemented `_get_current_confidence_threshold` in `SelfTrainingLightningModule` with linear decay (0.6 -> 0.25). Discovered and fixed a critical resumption bug in `BaseTrainer` and `scripts/train.py` where PyTorch Lightning was resetting the epoch counter to 0 upon restart, causing curriculum regression. Also implemented a safety stop in `DashboardTelemetryCallback` to prevent training overruns beyond `max_epochs`. Loop 55 achieved a peak of 81.86% PCK@0.2 at Epoch 35.
- **Absolute Priority**:
  1. **Record**: Loop 54 (**82.5% PCK@0.2**, **10.2 px MPJPE**) is the **new stable verified record**.
  2. **Next Goal**: Refine Adaptive Curriculum (Loop 56) or implement MoE Modality Routing.
- **Baseline**: Loop 54 (82.5% PCK@0.2).

## ⚠️ CRITICAL: Metric Audit Results

All previously reported PCK values in this tracker were computed by `scripts/evaluate.py` running on the **remote Kaggle environment** using **global default config** (not run-specific config), and using **soft-argmax for all models** regardless of training decoder. The numbers **cannot be trusted as absolute baselines**.

Fresh local re-evaluation established the following **corrected baselines** (cover1+cover2 val set, correct decoder per model):

| Run | Decoder | PCK@0.2 (strict) | MPJPE | Status |
|-----|---------|--------------------|--------------------|--------|
| **loop54_self_training_v3** | argmax | **82.5%** | **10.2 px** | **STABLE RECORD** |
| loop53_advanced_cover | argmax | **78.7%** | **11.9 px** | CNN/ViT Champion |
| **loop44_vitpose_fixed** | argmax | **77.8%** | **12.3 px** | Legacy Record |

## ⚠️ Scientific Caution: Loop 50 "Record"
While Loop 50 achieved a numeric peak of 78.41% (+0.57pp over Loop 44), this improvement is within the margin of error and variance. Given that CUT requires a pre-trained generator and adds significant data-pipeline complexity/latency, this approach currently **fails the cost-benefit analysis**. Future iterations must focus on *integrated* pose-consistent translation where the GAN actively improves skeletal localization.

## Iteration Log

| Loop ID | Hypothesis | Result | Corrected PCK@0.2 | Action |
|---------|-----------|--------|--------------|--------|
| 44 | Stabilized ViTPose Fine-tuning (Fixed Structure + Disc. LR) | SUCCESS | **77.8%** | Bypassing class token; backbone LR at 5e-6; stable sigma=3.0. **STABLE RECORD**. |
| 47 | Monochromatic CycleGAN Translation | SUCCESS | N/A | Enforced 1-ch output. Stable infra. |
| 48 | Contrastive Unpaired Translation (CUT) | SUCCESS | N/A | Applied Deep Semantic NCE Fix. Visual audit confirmed fabric hallucination. |
| 50 | ViTPose + CUT Augmentation | MARGINAL | **78.4%** | Numeric peak hit at E10, but fluctuated. Overhead is high. |
| 51 | Boosted CUT Augmentation (0.7) | FAILURE | 76.6% | **OVER-AUGMENTATION**: excessive domain noise. |
| 52 | Balanced Diversity (L44 + L49 CUT seasoning) | STALLED | 77.6% | Regained ground but confirmed diminishing returns of offline CUT augmentation. |
| 53 | Advanced Cover (FDA + HistMatch + Bank) | SUCCESS | **78.7%** | Refactored augmentations; dynamic reference bank improved realism. **NEW STABLE RECORD**. |
| 54 | Self-Training (EMA Teacher + CUT Strong Aug) | SUCCESS | **82.5%** | **v3 Run (Final Record)**: Achieved 82.5% PCK in a fresh, uninterrupted run. Verified the stability of EMA-based consistency regularization. |
| 55 | Adaptive Confidence Curriculum (0.6 -> 0.25) | MARGINAL | 81.86% | Decaying threshold increased target utilization but injected too much noise late-stage. Peak at E35. |

## Next Planned Steps (Post-Loop 54 Breakthrough)

1. **Adaptive Confidence Curriculum (Loop 55)**: Implement a decaying confidence threshold to maximize learning from difficult target domain samples as the teacher matures.
2. **MoE Modality Routing**: Route tokens between "Clean" and "Occluded" experts to handle capacity interference between source and target domains.
3. **Multi-View Consistency**: Leverage side views for cross-view pseudo-label regularization.
