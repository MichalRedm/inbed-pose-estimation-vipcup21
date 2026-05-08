# Ideas Log

## Hypothesis Queue
1. **Improved Occlusion Augmentation**: The current occlusion is just a polygon at the bottom. We need more realistic blanket simulations (textures, varying opacity, different shapes).
2. **Domain Adaptation (UDA)**: Since the goal is uncovered -> covered, we could use methods like entropy minimization or adversarial training to align features.
3. **Multi-Modal Pre-training**: If RGB is available for training but IR is used for covered, we can use RGB to help. But the user said Training (uncovered) -> Validation (covered).

## Web Research Syntheses
- **Synthetic Occlusion**: Pasting external objects or random masks over the subject forces the model to learn context.
- **SLP Dataset Specifics**: SLP features systematic cover (blankets). For IR, blankets act as insulators, diffusing and dampening the heat signature.
- **Multimodal Fusion**: Leveraging Pressure Maps or RGB can help, but since we are training on uncovered IR to predict covered IR, we need to bridge the domain gap via augmentation or UDA.

## Graveyard
- **Thermal Diffusion (Initial)**: Failed due to a critical rotation sign error in `src/data/augmentations.py` which corrupted training coordinates. Root cause: Image rotation was CCW but joint rotation was effectively CW.
- **Relative Data Paths (Remote)**: Failed on Kaggle GPU instances. Root cause: Execution context in Kaggle notebooks/scripts requires absolute paths for dataset discovery.

## Hypothesis Queue (Prioritized)
1. **Adversarial Domain Alignment**: Use a Discriminator to make the feature representations of "uncovered" (source) and "covered" (target) images indistinguishable.
2. **Joint-Aware Masking**: Randomly occlude specific limbs or the entire lower/upper body to simulate various blanket positions, combined with the fixed Thermal Diffusion.
3. **Multi-Scale Heatmap Regression**: Adjust the heatmap sigma based on joint type (larger sigma for lower body joints which are harder to localize under covers).
