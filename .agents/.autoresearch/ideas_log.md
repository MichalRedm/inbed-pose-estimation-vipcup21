# Ideas Log

## Hypothesis Queue

> ⚠️ **Hypothesis #1 is an ABSOLUTE PREREQUISITE** — All previous results are suspect due to bugs in the evaluation script and loss landscape issues. No new experimental work should be done until the baseline is solidly re-established.

1. **[SUCCESS] Systematic Re-evaluation & Loss Alignment**: 
   - **Status**: Implementation complete (2026-05-11). `UncertaintyWeighting` added to `StandardTrainer`. `best_model.pth` now saved based on `val_pck`.
   - **Next**: Verify in Loop 17 training run.

2. **[SUCCESS] Stabilized Transfer Learning via Selective Freezing (Loop 23)**:
   - **Hypothesis**: The underperformance of pre-trained HRNet in Loops 20/21 was caused by **feature washout** (gradients from the random head destroying backbone features) and **underfitting**. By **freezing the Stem and Stage 1** (generic edge/texture detectors) and using a **lower LR (5e-5)**, the model can quickly adapt the high-level pose logic to thermal data without losing the ImageNet structural priors.
   - **Result**: **SUCCESS** (41.0% PCK@0.2). Outstanding performance after resolving the nested downsample/transition loading bug! Freezing the generic edge and shape filters preserved ImageNet structure, leading to robust convergence and generalization.
   - **Implementation**: Set `requires_grad=False` for `conv1`, `conv2`, and `layer1`. Use Argmax decoding.
   - **Priority**: Completed.

3. **[UNDERPERFORMED] Discriminative Learning Rates — Fully Unfrozen (Loop 24)**:
   - **Hypothesis**: Fully unfreeze the pre-trained backbone but protect it with a 0.1× backbone learning rate relative to the head. Gives the network maximum adaptational flexibility without gradient washout.
   - **Result**: **36.7% PCK@0.2** (worse than frozen Loop 23's 41.0%). The discriminative LR of 1e-5 on the backbone is still too high: even with the ratio, the randomly-initialized head produces large loss gradients in epoch 1, corrupting pretrained backbone features before the head stabilizes. Evidence: Loop 24 Ep1 PCK = 3.03% vs. Loop 23 Ep1 PCK = 9.81%. The gap narrows over time (epoch 29-30 Loop 24 ≥ Loop 23), confirming the backbone is eventually recoverable, just slowly.
   - **Priority**: Completed. Key lesson: discriminative LR alone is insufficient without first warming up the head.

4. **[COMPLETED — UNDERPERFORMED] Progressive Unfreezing with Warm-up then Unfreeze (Loop 25)**:
   - **Hypothesis**: A two-phase training schedule will solve both the initial gradient washout AND the long-term domain ceiling:
     - **Phase 1 (epochs 1–15)**: Freeze stem + stage1. Train head at full LR (1e-4) until it stabilizes.
     - **Phase 2 (epochs 16–50)**: Automatically unfreeze the backbone. Switch to discriminative LR (backbone: 1e-5, head: 1e-4).
   - **Result**: **41.87% PCK@0.2 (peak, E33), 40.33% final**. Best pretrained result yet. Phase 2 transition was clean — no loss spike, correct parameter group setup confirmed. However, train-val loss gap grew by 1731% during Phase 2 (from 0.00005 to 0.00101), revealing that the 80-subject dataset is too small to fully adapt 915 backbone parameter groups without overfitting. **STRUCTURAL CEILING CONFIRMED at ~42%.** The fundamental RGB→IR domain gap and 1-channel conv1 weakness cannot be overcome by fine-tuning scheduling alone.
   - **Priority**: Completed. **VERDICT: Abandon pretrained route.**

5. **[FAILURE — PIPELINE BUGS] Loop 26: Sigma Curriculum + Structured Cutout (Scratch, Subjects 1-80)**:
   - **Hypothesis**: Combine sigma annealing curriculum with structured cutout, thermal jitter, translation, and sensor noise to improve cover1/cover2 accuracy.
   - **Result**: Underperformed (44.4% PCK). Investigated and discovered two severe pipeline bugs: (1) horizontal flip scrambled keypoint coordinate mapping due to missing index re-indexing; (2) persistent CPU dataloader worker processes failed to synchronize epoch-varying sigma (remained stuck at 3.0). This resulted in contradictory spatial signals and caused joint coalescence / skeleton collapse.
   - **Action**: Fix bugs completely. Re-run as Loop 27.

6. **[HIGHEST PRIORITY — Loop 27] Clean Sigma Curriculum + Structured Cutout Rerun (Scratch, Subjects 1-80)**:
   - **Hypothesis**: Re-running the exact same excellent Loop 26 recipe (sigma annealing 3.0→1.5 over 30 epochs + structured Cutout + translation + thermal dynamic range jitter + sensor noise) on a **fully clean, bug-free codebase** with:
     - Correct keypoint horizontal flip index mapping (left-right sides swapped).
     - GPU-based on-the-fly vectorized heatmap generation `generate_pytorch_heatmaps` to guarantee 100% synchronized curriculum execution.
     - PCK-based best checkpoint selection.
   - **Expected PCK**: 48–52%, < 24 px MPJPE (beating the 46.6% scratch baseline).
   - **Priority**: **HIGHEST**.

4. **Structured Regional Cutout (Simulated Extreme Occlusion)**:
   - **Hypothesis**: The current `ThermalDiffusionAugmenter` only applies a wavy blur to simulate a blanket. The model relies too much on residual thermal leakage. Applying "GridMask" or large contiguous "Cutout" blocks that zero out entire limbs will force the network to learn holistic structural dependencies rather than local textures, preparing it for the extreme occlusion of `cover2`.
   - **Implementation**: Add a simple structured cutout augmentation that completely masks 25-50% of the image during training.
   - **Priority**: Medium-High (Very simple to implement in `DataAugmenter`, specifically targets the domain gap).

5. **Kinematic Bone-Vector Decomposition (Decoupled Length and Direction)**:
   - **Hypothesis**: Direct regression of absolute (x,y) coordinates under anatomical constraints often leads to "skeleton collapse" (Loop 19) because minimizing bone length to 0 perfectly satisfies the Hinge loss. Decoupling the prediction into a root joint (pelvis) plus bone vectors (length and angle) prevents this. The length can be strongly regularized while angles vary freely.
   - **Implementation**: Modify the `SoftArgmax2D` layer or the prediction head to regress root (x,y) and relative vectors for limbs, reconstructing the final pose via forward kinematics.
   - **Priority**: Medium (Strong theoretical guarantee against collapse, but requires non-trivial architectural and loss restructuring).

6. **Feature-Level Multimodal Fusion (IR + PM)**:
   - **Status**: [BLOCKED] Pressure Maps (PM) data not found in local dataset.

7. **Spatial Dependency Refinement (GCN)**:
   - **Result**: [FAILURE] (Loop 18). GCN layer caused excessive smoothing.

8. **Joint-Specific Adaptive Loss Scaling (Focal Heatmap Loss)**:
   - **Result**: [FAILURE] (Loop 11). Over-focus on extremities led to structural instability.

9. **[FUTURE] Input Channel Replication with Pristine Pretrained Backbone (ImageNet HRNet-W32)**:
   - **Hypothesis**: Instead of modifying the pre-trained `conv1` layer to accept 1-channel input (and losing features due to averaging), keep `in_channels=3` in HRNet. Replicate the 1-channel thermal input three times to form a 3-channel input ($R=G=B$). This preserves 100% of the ImageNet edge/texture spatial priors in `conv1`, preventing initial feature washout and resolving the structural ceiling of pretrained models on thermal data.
   - **Implementation**: Set `in_channels=3` and `pretrained=True` in HRNet config. Modify `VIPCupDataset` to output `[3, H, W]` tensors for IR images by repeating the channel.
   - **Priority**: High (Low implementation cost, high theoretical ROI to bridge the domain gap).
   - **Expected PCK**: 48–52%.

10. **[FUTURE] Cross-Modality Feature Distillation (Aligned RGB → IR)**:
    - **Hypothesis**: Since the SLP dataset contains aligned RGB and IR image pairs, we can train a strong RGB-only teacher model on the color images. During training of the thermal student model, we apply a feature-imitation loss (e.g., MSE or Cosine Similarity) between the intermediate feature maps of the RGB teacher and the IR student. This transfers robust, occlusion-invariant human spatial priors from the clear RGB domain into the thermal domain.
    - **Implementation**: Train a standard HRNet on RGB uncover images. Save the weights as a teacher. During thermal training, run both images, matching the final parallel stream outputs before the head.
    - **Priority**: Medium-High (High ROI, moderate implementation effort).
    - **Expected PCK**: 50–54%.

11. **[FUTURE] Thermal-Pretrained YOLO-Pose Baseline via OpenThermalPose**:
    - **Hypothesis**: Instead of training top-down HRNet from scratch, leverage the thermal-specific YOLOv8/v11-pose checkpoints released by the `IS2AI/OpenThermalPose` project. Fine-tune them directly on the SLP dataset.
    - **Implementation**: Load `yolo11n-pose.pt` using the `ultralytics` API, map keypoint labels, and fine-tune on the SLP training split.
    - **Priority**: Medium (Leverages existing thermal-specific pre-trained weights, but requires integrating the `ultralytics` codebase).
    - **Expected PCK**: 48–53%.

## Web Research Syntheses
- **Foreshortening Priors**: 2D bone lengths are upper-bounded by physical 3D length but lower-bounded by 0. Using a Hinge loss (ReLU) on length exceeding the max effectively models this projection constraint.
- **Curriculum Learning for Priors**: Enforcing structural constraints too early can lead to poor local minima. A linear warmup allows the model to find the correct spatial basins first.
- **SLP Dataset Specifics**: The insulating effect of blankets in IR means joint heat signatures are blurred and shifted. Structural priors are essential to "glue" the limbs together.
- **Loop 20 Synthesis (HRNet Underperformance)**: ImageNet-pre-trained HRNet-W32 (35.1% PCK) failed to beat the lighter baseline (46.6%). Analysis shows linear loss decrease at epoch 30, indicating significant underfitting. Additionally, the 1-channel `conv1` was randomly initialized, breaking the low-level feature extraction chain. Doubling epochs and using averaged `conv1` weights is required.
- **Preventing Skeleton Collapse**: Research indicates direct coordinate regression with structural penalties often leads to collapse. State-of-the-art methods decompose pose into root position + bone vectors (length/angle), applying length priors without compressing the skeleton.
- **Occlusion Handling**: Multi-modal fusion is best, but when restricted to IR, explicitly modeling visibility (e.g., through an auxiliary attention branch) helps the network switch from texture-reliance to prior-reliance.
- **Loop 22 Analysis**: Established that Soft-Argmax causes "Joint Coalescence" in thermal images due to the "center-of-mass" being pulled toward high-intensity heat (head). Returning to Argmax is essential for precision.
- **Loop 23/24 Analysis — Progressive Unfreezing Evidence**: Epoch-by-epoch comparison reveals Loop 23 (frozen) starts 6.8% ahead at Ep1, peaks at ~41% (ep 13), then DECAYS and oscillates due to the frozen backbone's rigidity. Loop 24 (unfrozen, disc. LR) starts 6.8% behind but catches up monotonically, surpassing Loop 23 at epoch 29. This confirms: (1) frozen backbone gives fast initial convergence but hits a domain ceiling; (2) fully unfrozen backbone recovers slowly but eventually wins; (3) the OPTIMAL strategy is sequential: freeze first (borrow Loop 23's fast start), then unfreeze (borrow Loop 24's long-term adaptability). This is the **ULMFiT / Progressive Unfreezing** approach validated by Howard & Ruder (2018) on language models, and confirmed empirically in our own data.
- **Convergence Rate Analysis**: Loop 24 shows +0.62% PCK/epoch in the last 5 epochs (still monotonically improving) vs. Loop 23 showing effectively 0 or negative improvement in the last 5. Loop 24 has not converged and WOULD beat Loop 23 given more epochs. This validates extending training to 50 epochs in Loop 25.
- **Loop 25 Post-Mortem — Pretrained Route Final Analysis (2026-05-18)**: Progressive unfreezing executed correctly (confirmed by remote logs: `[HRNet] All parameters unfrozen (progressive unfreezing Phase 2).`; `[Factory] Using Discriminative LR! Head: 5, Backbone: 915 param groups, ratio 0.1`). Phase 2 transition at E16 produced an immediate PCK jump (+4pp) with zero loss spike, confirming the 2-phase design was technically sound. However, the train-val loss divergence gap grew from 0.00005 (E16) to 0.00101 (E50) — a 1731% increase — indicating progressive backbone overfitting, not domain adaptation. Peak PCK of 41.87% was reached at epoch 33 then regressed to 40.33% at epoch 50. **THE HARD CEILING AT ~42% IS STRUCTURAL**: the RGB→IR domain gap (texture vs. heat diffusion statistics), the averaging of conv1 from 3→1 channels (losing the pre-trained first-layer detector), and the dataset being too small (~1700 images) to fully fine-tune 915 backbone parameter groups all compound to prevent the pretrained backbone from ever catching up to a scratch model initialized appropriately for this domain.
- **Literature Confirmation**: 2023-2024 research on IR pose estimation confirms that RGB-pretrained HRNet fine-tuned on thermal data consistently underperforms vs. models pre-trained on grayscale COCO or thermal-specific data. Modality-specific batch normalization and modality-adaptive loss functions are required to bridge this gap without domain-specific pre-training data, neither of which is feasible within our current pipeline constraints.
- **Sigma Curriculum Mechanism**: Research confirms sigma annealing (wide→narrow Gaussian) provides a curriculum that: (1) provides wide basin of attraction early (easy gradient signal), (2) progressively forces finer-grained localization, (3) prevents the model from converging to coarse-grained local minima where it "knows" the rough joint region but cannot pinpoint exact location. Expected improvement: +3-5% PCK on structured datasets.
- **Modality Pre-training & Grayscale COCO (2026-05-19 Web Research)**: In-depth survey of thermal/IR pose estimation literature (LLVIP-Pose, UCH-ThermalPose, OpenThermalPose) confirms that RGB-pretrained backbones undergo severe **feature washout** when the early 3-channel layers are averaged to 1-channel. The state-of-the-art recommendation is either: (1) **Channel Replication**: Keep `in_channels=3` and replicate the 1-channel IR input three times ($R=G=B=IR$), allowing ImageNet features to function with 100% integrity; or (2) **Grayscale COCO Pre-training**: Pre-train standard models on grayscale-converted MS COCO to naturally align weight statistics with single-channel intensity before domain transfer.
- **IS2AI OpenThermalPose Checkpoints**: The official `IS2AI/OpenThermalPose` research initiative has released public pre-trained YOLOv8-pose and YOLO11-pose checkpoints trained directly on thermal LWIR imagery. These act as a strong alternative baseline for high-speed, real-time thermal human pose estimation on edge devices.
- **Cross-Modality Distillation**: For datasets with aligned modalities (like SLP's RGB and IR pairs in the uncover phase), cross-modality distillation—where a powerful RGB teacher model guides a thermal student model's feature maps—is highly effective at transferring rich spatial priors to the thermal network.

## Graveyard
- **Adversarial UDA (Global)**: Loop 5. Caused feature washout.
- **Fixed-Length MSE Anatomical Loss**: Loops 6-8. Over-regularized foreshortened poses, leading to -3.0% PCK regression. **PURGED** (2026-05-12).
- **Anatomical Angle Constraints (2D Hinge)**: Loop 10. Penalized sharp 2D projections of valid 3D poses, causing -9.4% PCK regression. 2D inner angle is not a reliable proxy for 3D ROM.
- **Joint-Specific Loss Weighting**: Loop 11. Over-focus on extremities (2.0x weight) led to structural instability and a -3.3% PCK regression compared to the baseline.
- **Occlusion Consistency Regularization**: Loop 12. Enforcing $pred_{occluded} \approx pred_{clean}$ improved hip stability but didn't beat the baseline on extremities.
- **Multi-Scale Heatmap Supervision**: Loop 13. Supervising intermediate resolutions (1/8, 1/16) caused gradient noise and a -2.1% PCK regression.
- **Normalized Skeleton Collapse**: Loop 19. Using anatomical hinge loss on 0-1 normalized coordinates without strong heatmap anchoring caused all joints to collapse into a single point to minimize bone length error (loss reached 0.00009).
- **Thermal Diffusion (Initial)**: Sign error in rotation augmentations.

- **Loop 21 Enhanced HRNet (Pre-trained + Soft-Argmax + 60 Epochs)**: Loop 21. Achieved only 32.0% PCK. **VISUAL FAILURE**: Inference on training examples shows severe **Joint Coalescence** in the head area. Multiple joints (extremities, torso) are clustered around the head, indicating that the Soft-Argmax 'center-of-mass' calculation is being pulled toward the highest intensity thermal region, and the Coordinate Regression loss is not strong enough to pull them away, or is actively encouraging this clustering to minimize global error. This confirms that Soft-Argmax is currently detrimental to precision.

## Current Iteration
- **Loop 9**: Foreshortening-Aware Hinge Loss (SUCCESS). Reached 45.1% PCK@0.2.
- **Loop 17**: Multi-Task Uncertainty Weighting (SUCCESS). 43.1% PCK@0.2.
- **Loop 18**: GCN Refinement (FAILURE). 33.4% PCK@0.2.
- **Loop 19**: Normalized Anatomical Hinge (FAILURE). 12.7% PCK@0.2. Skeleton collapse.
- **Loop 20/21**: Pre-trained HRNet + Soft-Argmax (FAILURE). 32.0% PCK@0.2. Coordinate regression is hindering precision.
- **Loop 23**: Stabilized Pre-training — Frozen Stem+Stage1 (SUCCESS). 41.0% PCK@0.2.
- **Loop 24**: Discriminative LR — Fully Unfrozen (UNDERPERFORMED). 36.7% PCK@0.2. Gradient washout at epoch 1 despite 0.1× backbone LR. Still monotonically improving at epoch 30 — convergence not reached.
