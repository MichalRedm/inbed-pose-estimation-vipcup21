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
*   **ROI Status**: **MEDIUM-HIGH (ROI Rank 2)** — High theoretical ROI for bridging the domain gap, but moderate-to-high implementation complexity.

### 3. Structured Regional Cutout (Simulated Extreme Occlusion)
*   **Hypothesis**: The current `ThermalDiffusionAugmenter` only applies a wavy blur to simulate a blanket. The model relies too much on residual thermal leakage through the sheets. Applying large contiguous "Cutout" blocks that zero out entire limbs during training forces the network to learn holistic structural dependencies rather than local textures, preparing it for the extreme occlusion of `cover2`.
*   **Implementation**: Add a simple structured cutout augmentation module in `src/data/augmentations.py` that dynamically masks 25-50% of the image during training.
*   **ROI Status**: **MEDIUM (ROI Rank 3)** — Very simple to implement, specifically targeting the lack of geometric reasoning in heavily occluded poses.

### 4. Thermal-Pretrained YOLO-Pose Baseline via OpenThermalPose
*   **Hypothesis**: Instead of training top-down HRNet from scratch, leverage the thermal-specific YOLOv8/v11-pose checkpoints released by the `IS2AI/OpenThermalPose` research initiative. Fine-tune them directly on the SLP dataset.
*   **Implementation**: Load `yolo11n-pose.pt` using the `ultralytics` API, map keypoint labels, and fine-tune on the SLP training split.
*   **ROI Status**: **MEDIUM (ROI Rank 4)** — High chance of working due to modality-aligned pre-training, but requires integrating the heavy external `ultralytics` package structure.

### 5. Kinematic Bone-Vector Decomposition (Decoupled Length and Direction)
*   **Hypothesis**: Direct regression of absolute (x,y) coordinates under anatomical constraints often leads to "skeleton collapse" because minimizing bone lengths to 0 perfectly satisfies the Hinge loss. Decoupling the prediction into a root joint (pelvis) plus bone vectors (length and angle) prevents this. The length can be strongly regularized while angles vary freely.
*   **Implementation**: Modify the model prediction head to regress root (x,y) and relative vectors for limbs, reconstructing the final pose via forward kinematics.
*   **ROI Status**: **LOW-MEDIUM (ROI Rank 5)** — Strong theoretical guarantee against collapse, but requires heavy non-trivial architectural and loss restructuring.

---

## 📝 Web Research Syntheses

*   **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
*   **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
*   **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.
*   **Preventing Skeleton Collapse**: Research indicates direct coordinate regression with structural penalties often leads to collapse. State-of-the-art methods decompose pose into root position + bone vectors (length/angle), applying length priors without compressing the skeleton.
*   **Occlusion Handling**: Multi-modal fusion is best, but when restricted to IR, explicitly modeling visibility (e.g., through an auxiliary attention branch) helps the network switch from texture-reliance to prior-reliance.
*   **Modality Pre-training & Grayscale COCO (2026-05-19 Web Research)**: In-depth survey of thermal/IR pose estimation literature (LLVIP-Pose, UCH-ThermalPose, OpenThermalPose) confirms that RGB-pretrained backbones undergo severe **feature washout** when the early 3-channel layers are averaged to 1-channel. The state-of-the-art recommendation is either **Channel Replication** ($R=G=B=IR$) or pre-training on grayscale-converted MS COCO to naturellement align weight statistics.

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

---
*Created and maintained under the Antigravity ML Autoresearch framework.*
