# Ideas Log

## Hypothesis Queue
1. **Improved Occlusion Augmentation**: The current occlusion is just a polygon at the bottom. We need more realistic blanket simulations (textures, varying opacity, different shapes).
2. **Domain Adaptation (UDA)**: Since the goal is uncovered -> covered, we could use methods like entropy minimization or adversarial training to align features.
3. **Multi-Modal Pre-training**: If RGB is available for training but IR is used for covered, we can use RGB to help. But the user said Training (uncovered) -> Validation (covered).

## Web Research Syntheses
- **Synthetic Occlusion**: Pasting external objects or random masks over the subject forces the model to learn context.
- **SLP Dataset Specifics**: SLP features systematic cover (blankets). For IR, blankets act as insulators, diffusing and dampening the heat signature.
- **Multimodal Fusion**: Leveraging Pressure Maps or RGB can help, but since we are training on uncovered IR to predict covered IR, we need to bridge the domain gap via augmentation or UDA.

## Hypothesis Queue (Prioritized)
1. **Heat Diffusion Augmentation**: Simulate the effect of blankets in IR by applying localized Gaussian blur and intensity reduction to the body regions in uncovered training images. This is more realistic than the current polygon-based occlusion.
2. **Adversarial Domain Alignment**: Use a Discriminator to make the feature representations of "uncovered" and "covered" images indistinguishable.
3. **Joint-Aware Masking**: Randomly occlude specific limbs or the entire lower/upper body to simulate various blanket positions.
