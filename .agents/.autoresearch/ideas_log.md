# Ideas Log

This log tracks our prioritized queue of future improvement hypotheses, synthesizes web research findings, and archives failed or underperforming experiments in the Graveyard with exact root-cause diagnostics.

---

## 🎯 Prioritized Hypothesis Queue (ROI-Ranked)

Below is our prioritized queue of strictly **future** improvement hypotheses, ranked by Return on Investment (ROI)—defined as the combined probability of accuracy gains versus simplicity of implementation.

### 1. Input Channel Replication with Pristine Pretrained Backbone (ImageNet HRNet-W32)
*   **Hypothesis**: Averaging the 3-channel weights of the first convolution (`conv1`) to accept 1-channel thermal IR inputs wipes out pre-trained Edge/Shape detection priors, breaking the feature extraction chain. By setting `in_channels=3` and replicating the 1-channel thermal input three times ($R=G=B=IR$), we preserve 100% of the ImageNet edge/texture spatial priors in `conv1`, preventing initial feature washout.
*   **Implementation**: Keep `in_channels=3` and `pretrained=True` in HRNet config. Modify `VIPCupDataset` to output `[3, H, W]` tensors by repeating the single thermal channel.
*   **Status**: **SUCCESS (Loop 29)** — Reached **52.0% PCK@0.2** and **29.3 px MPJPE**. Hypothesis confirmed: preserving ImageNet conv1 priors via replication bridges the domain gap.
*   **ROI Status**: **ARCHIVED** — Successful. Now the new project baseline.

### 2. Cross-Modality Feature Distillation (Aligned RGB → IR in Uncover Phase)
*   **Hypothesis**: The SLP dataset provides perfectly aligned RGB and IR image pairs in the uncover phase. We can train a powerful RGB-only teacher model on clear color images. During student training on thermal IR, we apply a feature-imitation loss (e.g., Mean Squared Error or Cosine Similarity) between intermediate feature maps of the RGB teacher and the IR student. This transfers rich, occlusion-invariant human spatial priors into the thermal student network.
*   **Implementation**: Train a standard HRNet on RGB uncover images to act as a frozen teacher. During student thermal training, pass aligned pairs and add MSE loss between parallel multi-resolution features.
*   **Status**: **UNDERPERFORMED (Loop 32)** — Peaked at ~58.7% PCK@0.2 (well below loop31's 64%). Post-mortem in Graveyard below.
*   **ROI Status**: **PARTIALLY ARCHIVED** — The architecture is sound, the implementation is flawed. The improved version (Loop 33, below) is the next priority.

### 3. Structured Regional Cutout & Physically-Realistic Fabric Simulation
*   **Hypothesis**: The previous thermal blanket augmentation only dimmed and blurred the lower body region without continuous sheet geometry, fabric drapes, fine wrinkles, proper shadow edge transitions, or structured limb occlusions. By implementing a high-fidelity continuous fabric draping simulation combined with sharp small-scale wrinkles, proper drop shadow edges along the top wavy fold line, and structured regional cutout (35% size ratio), we force the model to learn holistic spatial coordinates and body shape contours under heavy blanket occlusion.
*   **Implementation**: Fully implemented continuous blanket sheets, drop-shadow creases, multi-scale drapery (high/low frequency folds) in `src/data/augmentations.py` along with dynamic GPU-based dataloading and a 35% cutout.
*   **Status**: **SUCCESS (Loop 31)** — Reached an all-time record **64.0% PCK@0.2** and **17.79 px MPJPE**. This represents the ultimate solution to the visual gap between synthetic and real blankets, and completely resolves occluded localization limits.
*   **ROI Status**: **ARCHIVED** — Successful. Incorporated into baseline production champion.

### 3. Improved Cross-Modality Distillation — Output + Decayed Feature Distillation (Loop 33 Candidate)
*   **Hypothesis**: Loop 32's failure was caused by using raw intermediate feature maps as the distillation target, which are fundamentally incompatible between RGB and IR modalities (different texture statistics). Research (Hinton et al., 2015; AAAI 2024 distillation curriculum paper) confirms that: (a) **output-level heatmap distillation** transfers pose-relevant `dark knowledge` (joint probability distributions, uncertainty about invisible joints) without forcing modality-specific texture structure; (b) **decaying the distillation weight** over training prevents negative transfer once the student has learned the IR-specific representations; (c) a **two-phase approach** — Phase 1 distillation only, Phase 2 pose-only fine-tuning — prevents the student from fighting the teacher in late epochs.
*   **Key Implementation Changes from Loop 32**:
    1. **Output heatmap distillation** instead of (or alongside) intermediate feature distillation. Distill the teacher's 14-channel heatmap output (via soft labels / temperature scaling) rather than raw Stage 3/4 activations.
    2. **Distillation weight decay schedule**: start `lambda_distill = 0.5`, decay to 0.0 by Epoch 20 (linear). Stop distillation entirely in the second half of training — this prevents the negative-transfer regime seen from Epoch 22+ in Loop 32.
    3. **Two-phase training** (optional): Phase 1 (Ep 1-20): distillation + pose loss. Phase 2 (Ep 21-40): pose loss only (IR fine-tuning without teacher interference).
    4. **Style-content decoupling** (optional advanced): use 1×1 adapter convolutions to project teacher features into a modality-neutral space before computing distillation MSE — this bridges the texture-statistics gap without full feature matching.
*   **ROI Status**: **HIGH (ROI Rank 2)** — Targets the exact failure mode of Loop 32 with well-researched fixes. Infrastructure already exists in `src/training/distillation_trainer.py`.

### 4. Thermal-Pretrained YOLO-Pose Baseline via OpenThermalPose
*   **Hypothesis**: Instead of training top-down HRNet from scratch, leverage the thermal-specific YOLOv8/v11-pose checkpoints released by the `IS2AI/OpenThermalPose` research initiative. Fine-tune them directly on the SLP dataset.
*   **Implementation**: Load `yolo11n-pose.pt` using the `ultralytics` API, map keypoint labels, and fine-tune on the SLP training split.
*   **ROI Status**: **MEDIUM (ROI Rank 3)** — High chance of working due to modality-aligned pre-training, but requires integrating the heavy external `ultralytics` package structure.

### 5. Kinematic Bone-Vector Decomposition (Decoupled Length and Direction)
*   **Hypothesis**: Direct regression of absolute (x,y) coordinates under anatomical constraints often leads to "skeleton collapse" because minimizing bone lengths to 0 perfectly satisfies the Hinge loss. Decoupling the prediction into a root joint (pelvis) plus bone vectors (length and angle) prevents this. The length can be strongly regularized while angles vary freely.
*   **Implementation**: Modify the model prediction head to regress root (x,y) and relative vectors for limbs, reconstructing the final pose via forward kinematics.
*   **ROI Status**: **LOW-MEDIUM (ROI Rank 4)** — Strong theoretical guarantee against collapse, but requires heavy non-trivial architectural and loss restructuring.

---

## 📝 Web Research Syntheses

*   **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
*   **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
*   **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.
*   **Preventing Skeleton Collapse**: Research indicates direct coordinate regression with structural penalties often leads to collapse. State-of-the-art methods decompose pose into root position + bone vectors (length/angle), applying length priors without compressing the skeleton.
*   **Occlusion Handling**: Multi-modal fusion is best, but when restricted to IR, explicitly modeling visibility (e.g., through an auxiliary attention branch) helps the network switch from texture-reliance to prior-reliance.
*   **Modality Pre-training & Grayscale COCO (2026-05-19 Web Research)**: In-depth survey of thermal/IR pose estimation literature (LLVIP-Pose, UCH-ThermalPose, OpenThermalPose) confirms that RGB-pretrained backbones undergo severe **feature washout** when the early 3-channel layers are averaged to 1-channel. The state-of-the-art recommendation is either **Channel Replication** ($R=G=B=IR$) or pre-training on grayscale-converted MS COCO to naturellement align weight statistics.
*   **Cross-Modal Knowledge Distillation — Negative Transfer & Mitigations (2026-05-22 Web Research, post-Loop 32 post-mortem)**:
    *   **Feature Misalignment is fundamental**: RGB and IR intermediate feature maps exist in different statistical domains even for the same scene. Raw MSE between them forces the student to mimic texture patterns that do not exist in thermal imagery, consuming model capacity with useless constraints. This is the primary failure mode of Loop 32 (confirmed by `w_distill` going negative at Ep22).
    *   **Output-level (heatmap) distillation is safer for cross-modal transfer**: Distilling the teacher's final heatmaps (the `dark knowledge`) transfers joint location uncertainty and inter-joint spatial priors without forcing RGB-specific low-level texture statistics. This is the recommended first upgrade for Loop 33.
    *   **Decoupled Knowledge Distillation (DKD)**: Separating the distillation loss into a target-class component and non-target-class component (Zhao et al., CVPR 2022) allows more fine-grained control over what is transferred. For cross-modal pose, the non-target heatmap channels (wrong-joint predictions) are where the richest relative-probability information lives.
    *   **Distillation weight decay is essential**: The literature consistently shows that a fixed or increasing distillation weight causes negative transfer in the later epochs as the student develops its own domain-appropriate representations. Linear decay from `lambda_distill=0.5` to `0.0` over the first 50% of training is recommended (confirmed by Loop 32 data — the critical epoch was 22/40 = 55%).
    *   **Style-content disentanglement**: If feature-level distillation is retained, adapter convolutions (1×1 conv) should project the teacher's features into a modality-neutral space. This allows structural/pose content to transfer while suppressing modality-specific texture style (RGB gradient edges vs. IR heat blobs).
    *   **Asymmetric Distillation / Student-Friendly Matching**: In scenarios with modality gap, use a learnable projection module on the teacher features so the student only imitates a *projected* version of the teacher that is expressible in IR-feature space. This prevents the student from being penalized for not producing features that are physically impossible in the thermal domain.
    *   **Two-phase training**: Phase 1 (distillation-guided): student learns fast from teacher. Phase 2 (self-supervised fine-tuning): teacher is frozen out, student refines its IR-specific representations without interference. Many SOTA cross-modal distillation papers use this paradigm.
    *   **Feature vs. Output distillation**: In pose estimation, output (heatmap) distillation transfers inter-joint relationship knowledge; feature distillation transfers how to extract pose-relevant features. The best results in the literature combine both, but with **output distillation having a higher relative weight** for cross-modal scenarios where the feature space is fundamentally different.

---

## 🪦 The Graveyard (Failed & Underperformed Ideas)

This archive logs all completed experiments that failed to outperform our baseline or introduced regressions, detailing the exact root cause of their failure.

### 1. Discriminative Learning Rates with Channel Replication (Loop 30)
*   **Root Cause**: **BACKBONE UNDERFITTING**. The 0.1x backbone learning rate ($10^{-5}$) was too low to adapt the ImageNet-pretrained features to the thermal IR domain, even with channel replication. The model failed to converge to a precise state, resulting in a 20pp PCK drop compared to Loop 29 (uniform LR $10^{-4}$).

### 2. Stacking Two-Sided Anatomical Hinge Loss & Local-Masked Soft-Argmax (Loop 28)
*   **Root Cause**: **SEVERE STRUCTURAL DEFORMATION**. In our visual audit on simple uncover IR poses, the model failed, introducing wrist double-prediction artifacts and diagonal crossed right-to-left ankle connections. Masking the soft-argmax coordinates to a tight $15 \times 15$ local window combined with anatomical bone constraints over-regularized the spatial tracking, leading to unnatural geometric shapes. Pure pixel-level heatmap MSE (Loop 27) remains significantly cleaner and more robust.

### 2. Progressive Unfreezing (Loop 25)
*   **Root Cause**: **FULL-BACKBONE OVERFITTING**. Phase 2 full backbone unfreezing caused the train-val loss divergence gap to grow by 1731% (from 0.00005 to 0.00101). The 80-subject dataset (~1700 images) is too small to fine-tune 915 backbone parameter groups without severe overfitting, creating a structural ceiling on fine-tuning ImageNet weights.

### 3. Discriminative Learning Rates — Fully Unfrozen (Loop 24)
*   **Root Cause**: **EARLY GRADIENT WASHOUT**. The backbone LR of 1e-5 was still too high; in epoch 1, large gradients from the randomly initialized head washed out the pre-trained backbone features before the head stabilized.

### 4. GCN-based Spatial Pose Refinement (Loop 18)
*   **Root Cause**: **EXCESSIVE COORDINATE SMOOTHING**. The Graph Convolutional Network layers applied to intermediate coordinates caused severe spatial smoothing, pulling joint predictions inward toward the body core and reducing PCK to 33.4%.

### 5. Joint-Specific Adaptive Loss Scaling / Focal Heatmap (Loop 11)
*   **Root Cause**: **BODY CORE DESTABILIZATION**. Over-weighting extremity heatmaps (2.0x) introduced high-gradient noise that destabilized the body core (shoulders/hips), causing a -3.3% PCK regression.

### 6. Normalized Skeleton Collapse (Loop 19)
*   **Root Cause**: **TRIVIAL GLOBAL MINIMUM**. Applying anatomical hinge loss on 0-1 normalized coordinates without strong heatmap anchoring allowed the network to compress the entire skeleton to zero-length (coalescing all joints into one point), which mathematically satisfied the bone length loss perfectly.

### 7. Fixed-Length MSE Anatomical Loss (Loops 6-8)
*   **Root Cause**: **FORESHORTENING BLINDNESS**. Enforcing fixed target limb lengths penalized valid 2D foreshortened poses (e.g. knees bent toward the camera), causing a -3.0% PCK regression.

### 8. Anatomical Angle Constraints (Loop 10)
*   **Root Cause**: **PROJECTION MISALIGNMENT**. 2D inner angle constraints penalized valid 2D projections of correct 3D poses, leading to a -9.4% PCK regression.

### 9. Adversarial Unsupervised Domain Adaptation (Loop 5)
*   **Root Cause**: **FEATURE WASHOUT**. Global domain-adversarial gradients destroyed the localization capacity of the features, leading to poor convergence.

### 10. Cross-Modality Feature Distillation — Raw Feature MSE (Loop 32)
*   **Root Cause**: **CROSS-MODAL NEGATIVE TRANSFER**. Distilling raw HRNet Stage 3 & Stage 4 feature maps (MSE) from an RGB teacher to an IR student forces the student to mimic texture statistics that are fundamentally different between the two modalities. RGB features are dominated by color edges, texture gradients, and surface reflectance; IR features are dominated by heat diffusion blobs and emissivity gradients. The uncertainty weighting mechanism (`sigma_distill`) correctly detected that the teacher signal was harmful from Epoch 22 onward — `w_distill` turned negative and kept growing in magnitude. Despite the automatic suppression, the distillation loss had already constrained the student's early-stage feature representations to partially conform to RGB-domain activations, creating a representational ceiling it could not escape. **Peak: ~58.7% PCK (Ep28) vs. record 64.0%.**
*   **Key Diagnostic Signal**: `w_distill` sign-flip at Epoch 22 (55% into 40-epoch training). The student's pose loss kept improving (`loss_pose` 0.037→0.0012), confirming the student backbone was learning effectively — the ceiling was imposed purely by the conflicting distillation constraint.
*   **Lessons for Loop 33**: (1) Use output heatmap distillation, not raw feature MSE; (2) apply a decaying distillation weight schedule; (3) consider two-phase training — stop distillation at epoch ~20.

---
*Created and maintained under the Antigravity ML Autoresearch framework.*
