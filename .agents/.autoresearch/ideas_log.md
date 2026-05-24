# Ideas Log

This log tracks our prioritized queue of future improvement hypotheses, synthesizes web research findings, and archives failed or underperforming experiments in the Graveyard with exact root-cause diagnostics.

---

## 🎯 Prioritized Hypothesis Queue (ROI-Ranked)

Below is our prioritized queue of strictly **future** improvement hypotheses, ranked by Return on Investment (ROI)—defined as the combined probability of accuracy gains versus simplicity of implementation.

### 1. ViTPose (Plain ViT-B Backbone with Classic Upsampling Decoder)
*   **Hypothesis**: Transitioning from a hierarchical convolutional backbone (HRNet) to a plain, non-hierarchical Vision Transformer (ViT-B-16) will resolve the occlusion ceiling. Because a plain ViT processes the image via patch embeddings and applies global self-attention across all patches from the very first layer, it naturally learns long-range anatomical dependencies. If a joint (like a wrist or ankle) is completely occluded under a blanket, the global self-attention layers can implicitly infer its position by attending to confident visible anchors (shoulders, hips, head) across the entire image. This completely eliminates lossy 1D coordinate bottlenecks and flat-heatmap grid shifting no-ops.
*   **Implementation**: Use PyTorch-native `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)` with input channel replication. Feed the backbone's 1D spatial output sequence (reshaped to $16 \times 16$ grid) into a lightweight classic decoder: two consecutive transpose convolutions (`ConvTranspose2d` with kernel size 4, stride 2, padding 1) to upsample to $64 \times 64$, followed by a final convolution to predict the 14 joint heatmaps.
*   **ROI Status**: **EXTREMELY HIGH (ROI Rank 1)** — Simple, powerful transition that resolves both the early convolutional feature washout and the spatial coordinate bottleneck of JSSCA.

### 2. Thermal-Pretrained YOLO-Pose Baseline via OpenThermalPose
*   **Hypothesis**: Instead of training top-down networks from scratch, leverage the thermal-specific YOLOv8/v11-pose checkpoints released by the `IS2AI/OpenThermalPose` research initiative. Fine-tune them directly on the SLP dataset.
*   **Implementation**: Load `yolo11n-pose.pt` using the `ultralytics` API, map keypoint labels, and fine-tune on the SLP training split.
*   **ROI Status**: **MEDIUM (ROI Rank 2)** — High chance of working due to modality-aligned pre-training, but requires integrating the heavy external `ultralytics` package structure.

### 3. Grayscale COCO Pre-training
*   **Hypothesis**: RGB-pretrained networks suffer from feature washout because the first-layer filters are optimized for color edges. Pre-training the backbone on grayscale-converted MS COCO before pose fine-tuning aligns weight statistics, creating robust monochromatic priors.
*   **Implementation**: Convert COCO images to grayscale and pre-train the ViT or HRNet backbone on COCO keypoints before SLP fine-tuning.
*   **ROI Status**: **MEDIUM (ROI Rank 3)** — High theoretical value, but requires large-scale dataset pipeline engineering.

---

## 📝 Web Research Syntheses

*   **ViTPose & ViTPose++ (2026-05-24 Web Research)**:
    - **Global Attention**: Standard vision transformers process the input image as a sequence of $16 \times 16$ patches and perform global self-attention. This global receptive field is incredibly effective at handling heavy occlusions, allowing the model to naturally infer the position of hidden joints (wrists/ankles under blankets) based on all other patches (shoulders/hips/head) across the entire frame.
    - **Lightweight Decoders**: Research shows that a plain ViT backbone paired with a lightweight classic decoder (two transposed convolutions upsampling to $64 \times 64$) achieves state-of-the-art keypoint localization, bypassing complex neck/attention networks.
    - **Grayscale Modality Transfer**: Converting RGB datasets to grayscale for pre-training prevents domain feature washout and matches the statistics of infrared (thermal) imagery perfectly.
*   **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
*   **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
*   **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.
*   **Preventing Skeleton Collapse**: Research indicates direct coordinate regression with structural penalties often leads to collapse. State-of-the-art methods decompose pose into root position + bone vectors (length/angle), applying length priors without compressing the skeleton.
*   **Occlusion Handling**: Multi-modal fusion is best, but when restricted to IR, explicitly modeling visibility (e.g., through an auxiliary attention branch) helps the network switch from texture-reliance to prior-reliance.
*   **Modality Pre-training & Grayscale COCO (2026-05-19 Web Research)**: In-depth survey of thermal/IR pose estimation literature (LLVIP-Pose, UCH-ThermalPose, OpenThermalPose) confirms that RGB-pretrained backbones undergo severe **feature washout** when the early 3-channel layers are averaged to 1-channel. The state-of-the-art recommendation is either **Channel Replication** ($R=G=B=IR$) or pre-training on grayscale-converted MS COCO to naturellement align weight statistics.
*   **Cross-Modal Knowledge Distillation — Negative Transfer & Mitigations (2026-05-22 Web Research, post-Loop 32 post-mortem)**:
     - Even with output-level heatmap distillation (KL divergence) and linear decay, cross-modality distillation from RGB to IR fundamentally conflicts with physical occlusion augmentations (e.g., synthetic blankets). The RGB teacher models clear, uncovered pose distributions perfectly. When the student is fed heavily occluded IR images (simulated blankets) and forced to mimic the teacher's confident, clear-vision predictions, the student fails to learn the uncertainty and physical properties of the occlusion. It is effectively penalized for behaving like a thermal model under occlusion. SOTA cross-modal distillation is effective *only* when both modalities share similar occlusion states during training.

---

## 🪦 The Graveyard (Failed & Underperformed Ideas)

This archive logs all completed experiments that failed to outperform our baseline or introduced regressions, detailing the exact root cause of their failure.

### 1. ViTPose from scratch / ImageNet-Pretrained (Loop 42)
*   **Root Cause**: **THE SPATIAL RESOLUTION BOTTLENECK**: Standard ViT-B-16 immediately downsamples the 256x256 image into a 16x16 grid of patches. For human pose estimation (a dense prediction task), a 16x16 feature map is extremely coarse, losing crucial high-frequency spatial details required for precise limb localization.
*   **Root Cause**: **LACK OF INDUCTIVE BIAS & DATA HUNGER**: Unlike CNNs, plain Transformers lack local translation invariance and must learn spatial relationships from scratch. While SOTA ViTPose performs extremely well on massive datasets (COCO/AIC), our 80-subject dataset is too small to teach a plain ViT-B how to route spatial coordinate information globally. The model achieved a peak of only 41.75% PCK, well below our 64.3% baseline.
*   **Lesson**: To use Vision Transformers successfully on small datasets, we must load weights pre-trained on a massive pose estimation dataset (like MS COCO) where the model has already learned correct global spatial routing rules.

### 2. JSSCA-v6 Confidence-Gated Spatially-Anchored Attention (Loop 41)
*   **Root Cause**: **GRID SHIFTING OCCLUSION CEILING**: Under extreme blanket occlusions (duvets), the backbone's heatmaps are flat ($conf \approx 0$). Differentiably shifting flat heatmaps using `F.grid_sample` is a no-op, forcing the network to reconstruct the peak from scratch via a small MLP residual decoder.
*   **Root Cause**: **FALLBACK TO TRAINING CENTROIDS**: Because the coordinate anchor was zeroed out by confidence gating (to isolate noise), the MHA layer had no spatial reference for occluded extremities. The MLP was forced to fallback to predicting a static peak at the global training average (inside the torso core), causing wrists and elbows to collapse inside the body midline.
*   **Lesson**: Post-processing coordinate regression on lossy 1D tokens is physically limited under zero visibility. We must perform global attention in a dense spatial representation space (such as a plain ViT backbone or dense spatial feature neck) to preserve spatial skip-connections and location-aware receptive fields.

### 2. JSSCA-v5 Spatially-Anchored Attention Post-Processor (Loop 40)
*   **Root Cause**: **COORDINATE ANCHOR POLLUTION UNDER EXTREME OCCLUSION**: For heavily occluded extremity joints (ankles/wrists) under blankets, the HRNet backbone outputs flat, blurred, or noisy heatmaps. Performing `soft_argmax_2d` on these uncertain/flat heatmaps generates highly chaotic coordinate anchors (pulled to the center `(0, 0)`). Gating these noisy coordinates with a simple dense coordinate encoder projected this chaos into the 256-dimensional joint tokens, **polluting** the self-attention layer and confusing the physical geometric reasoning of other limb joints (causing a 15–20% regressive drop on ankles/wrists down to ~34%, and elbows/knees down to ~40-45%).
*   **Root Cause**: **SOFT-ARGMAX DECODING COLLAPSE**: Standard `soft-argmax` decoding on heatmaps produced by JSSCA-v5 collapsed PCK to 3.3%. Because the model was trained with standard heatmap MSE, the output activations have arbitrary ranges (including negative values) representing a Gaussian shape but are not constrained probability distributions. Softmax-temperature scaling on these unnormalized ranges created severe edge-noise sensitivity, driving expected values to boundary centroids.
*   **Lesson**: To prevent noisy coordinate anchors from polluting joint tokens, the coordinate encoder must be gated by the backbone's peak confidence score (e.g. `conf = heatmaps.view(B, J, -1).max(dim=-1)[0]`). When confidence is extremely low, the coordinate anchor must be completely suppressed/ignored.
*   **Lesson**: Models trained with heatmap MSE must be decoded using pure `argmax` peak detection unless explicitly trained with a differentiable soft-argmax coordinate loss.

### 2. JSSCA-v4 with intermediate U-Net Skips (Loop 39 - collapsed run)
*   **Root Cause**: **DEGENERATE GRADIENT SHORTCUT / BYPASS OF ATTENTION BOTTLENECK**: Adding intermediate skip connections (`d1 = d1 + h3`, `d2 = d2 + h2`) directly from the joint-wise encoder to the progressive deconvolutional decoder created a shallow CNN shortcut. Gradients flowed completely through this shortcut, bypassing the 14-joint self-attention bottleneck. Since the shortcut was joint-wise (lacking geometric context), the network learned a degenerate noisy identity mapping. This flooded the output heatmaps with high-frequency noise, washing out the pre-trained backbone features in the first epochs and collapsing PCK to 2.1%.
*   **Lesson**: Avoid intermediate skip connections in post-processing attention blocks. To preserve sub-pixel peaks without creating degenerate shortcuts, project the coordinated tokens directly to 8x8 space and upsample progressively using a pure convolutional decoder without skips.

### 2. JSSCA-v3 Spatial Tokenization Post-Processor (Loop 38)
*   **Root Cause**: **VISUAL/SPATIAL TOKEN DILUTION & SPARSE BACKGROUND FLOODDING**: Partitioning each joint's heatmap into an $8\times 8$ spatial token grid yielded a sequence length of `14 * 64 = 896` tokens. Because keypoint heatmaps are extremely sparse (almost entirely zero except for a tiny Gaussian peak), 99% of these tokens represent empty background space. The Self-Attention layer was completely flooded with background noise, diluting the joint semantic identity and washing out the local peak features. Furthermore, downsampling sparse heatmaps through three consecutive strided convolutions with `stride=2` caused local activations to vanish before reaching the attention bottleneck, causing a catastrophic coordinate collapse down to **26.8% PCK@0.2**.
*   **Lesson**: Joint self-attention must operate on a highly compact representation (sequence length equal to the 14 joint identities) to prevent token dilution. Precise coordinates should be preserved via multi-scale skip connections rather than spatial token sequences.

### 2. JSSCA-v2 Stabilized Neck Attention (Loop 37)
*   **Root Cause**: **LATENT REPRESENTATION DRIFT & SPATIAL BLUR**: Inserting the attention block in the high-dimensional backbone representation space `(B, 480, 64, 64)` *before* the output head forces the model to learn joint detection and joint coordination simultaneously in a dense latent space. Lacking explicit joint semantic identity anchors (which JSSCA-v1 had by operating directly on 14-channel keypoint heatmaps), the transformer learned non-anatomical visual correlations (matching joints to blanket texture folds). Furthermore, bottleneck downsampling to $8\times 8$ followed by deconvolutional reconstruction introduced spatial blur and coordinate drift, capping accuracy at **63.6% PCK@0.2**, failing to match our post-processing baseline of **66.56%**.
*   **Lesson**: Post-processing attention is structurally superior for joint coordination because operating directly in keypoint heatmap space preserves explicit joint semantic identity anchors, allowing the transformer to focus 100% of its capacity on anatomical priors and geometric corrections.

### 2. JSSCA-v2 Neck Attention without Normalization or FFN (Loop 36)
*   **Root Cause**: **SEVERE GRADIENT & ACTIVATION EXPLOSION**: The raw `MultiheadAttention` layer was inserted in `JointSpatialChannelAttention` without pre-LayerNorm, post-attention LayerNorm, FFN blocks, or proper residual paths. During training, backpropagated features blew up exponentially, driving validation loss from `0.0013` up to `11.35` (Epoch 37) and finally `378263.28` (Epoch 40).
*   **Root Cause**: **MODEL COLLAPSE & SKELETON DRIFT**: The numerical instability caused extreme coordinate predictions and skeleton drift. PCK@0.2 degraded from an intermediate `65.5%` peak (Epoch 30) down to a final `63.3%` PCK@0.2, failing to match our baseline `66.56%`.
*   **Lesson**: Transformer layers must always incorporate robust Pre-LN normalization paths and Feed-Forward Network blocks to stabilize backpropagation and feature refinement, especially when training complex multi-resolution architectures.

### 2. Kinematic Bone-Vector Decomposition (Loop 34)

*   **Root Cause**: **CUMULATIVE ERROR PROPAGATION**: Decoupling coordinates into a recursive kinematic tree propagates errors down the limbs. In coordinate space, the prediction of a distal leaf (wrist/ankle) depends on a long chain of limb direction and scaling offsets from the root (neck/pelvis). Angular and length errors accumulate at each bone junction, causing massive offset drift for wrists and ankles (PCK@0.2 drops to 8-16% on extremities).
*   **Root Cause**: **COORDINATE SMOOTHING FROM SOFT-ARGMAX**: Kinematic reconstruction takes soft-argmax coordinates as input. Soft-argmax's mathematical expectation over heatmaps inherently acts as a spatial smoothing operator, which pulls joint predictions toward the body's centroid and dampens geometric variance, making it highly susceptible to systemic joint shift. High-resolution pixel-level heatmap argmax peak detection remains far more precise.

### 2. Improved Cross-Modality Distillation — Output Heatmap Distillation (Loop 33)
*   **Root Cause**: **SUPERVISION CONFLICT WITH OCCLUSION PHYSICS**. Distilling output heatmaps from an RGB teacher (trained on clear uncover images) to an IR student (trained on uncover images but heavily augmented with synthetic thermal blankets) actively hurts the student. The teacher confidently predicts joint locations based on RGB edges, but the student needs to learn the physical diffusion and blur of thermal energy through a blanket. Forcing the student to match the teacher's sharp RGB-based distributions prevents the student from learning the true thermal occlusion mapping. Result: 56.2% PCK@0.2 vs. 64.0% baseline.

### 3. Discriminative Learning Rates with Channel Replication (Loop 30)
*   **Root Cause**: **BACKBONE UNDERFITTING**. The 0.1x backbone learning rate ($10^{-5}$) was too low to adapt the ImageNet-pretrained features to the thermal IR domain, even with channel replication. The model failed to converge to a precise state, resulting in a 20pp PCK drop compared to Loop 29 (uniform LR $10^{-4}$).
*   **Root Cause**: **SEVERE STRUCTURAL DEFORMATION**. In our visual audit on simple uncover IR poses, the model failed, introducing wrist double-prediction artifacts and diagonal crossed right-to-left ankle connections. Masking the soft-argmax coordinates to a tight $15 \times 15$ local window combined with anatomical bone constraints over-regularized the spatial tracking, leading to unnatural geometric shapes. Pure pixel-level heatmap MSE (Loop 27) remains significantly cleaner and more robust.

### 4. Progressive Unfreezing (Loop 25)
*   **Root Cause**: **FULL-BACKBONE OVERFITTING**. Phase 2 full backbone unfreezing caused the train-val loss divergence gap to grow by 1731% (from 0.00005 to 0.00101). The 80-subject dataset (~1700 images) is too small to fine-tune 915 backbone parameter groups without severe overfitting, creating a structural ceiling on fine-tuning ImageNet weights.

### 5. Discriminative Learning Rates — Fully Unfrozen (Loop 24)
*   **Root Cause**: **EARLY GRADIENT WASHOUT**. The backbone LR of 1e-5 was still too high; in epoch 1, large gradients from the randomly initialized head washed out the pre-trained backbone features before the head stabilized.

### 6. GCN-based Spatial Pose Refinement (Loop 18)
*   **Root Cause**: **EXCESSIVE COORDINATE SMOOTHING**. The Graph Convolutional Network layers applied to intermediate coordinates caused severe spatial smoothing, pulling joint predictions inward toward the body core and reducing PCK to 33.4%.

### 7. Joint-Specific Adaptive Loss Scaling / Focal Heatmap (Loop 11)
*   **Root Cause**: **BODY CORE DESTABILIZATION**. Over-weighting extremity heatmaps (2.0x) introduced high-gradient noise that destabilized the body core (shoulders/hips), causing a -3.3% PCK regression.

### 8. Normalized Skeleton Collapse (Loop 19)
*   **Root Cause**: **TRIVIAL GLOBAL MINIMUM**. Applying anatomical hinge loss on 0-1 normalized coordinates without strong heatmap anchoring allowed the network to compress the entire skeleton to zero-length (coalescing all joints into one point), which mathematically satisfied the bone length loss perfectly.

### 9. Fixed-Length MSE Anatomical Loss (Loops 6-8)
*   **Root Cause**: **FORESHORTENING BLINDNESS**. Enforcing fixed target limb lengths penalized valid 2D foreshortened poses (e.g. knees bent toward the camera), causing a -3.0% PCK regression.

### 10. Anatomical Angle Constraints (Loop 10)
*   **Root Cause**: **PROJECTION MISALIGNMENT**. 2D inner angle constraints penalized valid 2D projections of correct 3D poses, leading to a -9.4% PCK regression.

### 11. Adversarial Unsupervised Domain Adaptation (Loop 5)
*   **Root Cause**: **FEATURE WASHOUT**. Global domain-adversarial gradients destroyed the localization capacity of the features, leading to poor convergence.

### 12. Cross-Modality Feature Distillation — Raw Feature MSE (Loop 32)
*   **Root Cause**: **CROSS-MODAL NEGATIVE TRANSFER**. Distilling raw HRNet Stage 3 & Stage 4 feature maps (MSE) from an RGB teacher to an IR student forces the student to mimic texture statistics that are fundamentally different between the two modalities. RGB features are dominated by color edges, texture gradients, and surface reflectance; IR features are dominated by heat diffusion blobs and emissivity gradients. The uncertainty weighting mechanism (`sigma_distill`) correctly detected that the teacher signal was harmful from Epoch 22 onward — `w_distill` turned negative and kept growing in magnitude. Despite the automatic suppression, the distillation loss had already constrained the student's early-stage feature representations to partially conform to RGB-domain activations, creating a representational ceiling it could not escape. **Peak: ~58.7% PCK (Ep28) vs. record 64.0%.**
*   **Key Diagnostic Signal**: `w_distill` sign-flip at Epoch 22 (55% into 40-epoch training). The student's pose loss kept improving (`loss_pose` 0.037→0.0012), confirming the student backbone was learning effectively — the ceiling was imposed purely by the conflicting distillation constraint.
*   **Lessons for Loop 33**: (1) Use output heatmap distillation, not raw feature MSE; (2) apply a decaying distillation weight schedule; (3) consider two-phase training — stop distillation at epoch ~20.

---
*Created and maintained under the Antigravity ML Autoresearch framework.*
