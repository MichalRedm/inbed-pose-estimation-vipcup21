# Ideas Log

## Stochastic CUT (BicycleCUT / Mode-Seeking CUT)
**Date:** 2026-06-05
**Status:** Proposed

### Motivation
The standard CUT (Contrastive Unpaired Translation) model is fully deterministic. It maps one uncovered image to exactly one covered image. This is a limitation compared to synthetic augmentation, where we can generate multiple different randomized variations (covers) for the same underlying structure (pose).

### Proposed Idea
Modify the CUT architecture to become a stochastic generator by injecting a random noise vector `z` during the forward pass. This allows generating slightly different covers (e.g., different blanket wrinkles, textures, or lighting) for the same uncovered image by sampling different values of `z`.

### Implementation Details
To achieve this, we can borrow concepts from BicycleGAN, StarGAN v2, and Mode Seeking GANs:

1. **Noise Injection (Generator Modification):**
   - **Concatenation:** Expand the noise vector `z` spatially and concatenate it with the input image `x` or intermediate feature maps.
   - **AdaIN (Adaptive Instance Normalization):** Instead of standard InstanceNorm, use AdaIN in the ResNet blocks of the generator. An MLP maps the noise vector `z` to the affine parameters (scale and shift) of the AdaIN layers, effectively controlling the "style" of the generated cover.

2. **Preventing Mode Collapse:**
   Standard I2I generators often learn to ignore concatenated noise vectors. To force the generator to utilize `z`, we need an additional regularization loss:
   - **Mode Seeking Loss (MS-Loss):** Add a penalty that maximizes the distance between generated images relative to the distance between their corresponding latent vectors: `L_ms = ||G(x, z1) - G(x, z2)|| / ||z1 - z2||`.
   - **Latent Regression (BicycleGAN approach):** Introduce a small Encoder network `E` that attempts to recover `z` from the generated image: `L_z = ||E(G(x, z)) - z||_1`.

3. **Compatibility with CUT (PatchNCE):**
   The PatchNCE loss in CUT enforces structural consistency between the input and the generated image. This works perfectly with the proposed stochasticity: the PatchNCE loss will ensure the underlying body pose remains consistent with the input, while the injected noise `z` will manipulate the appearance/style of the generated cover, providing the desired randomized augmentation.

### Conclusion
This is a highly reasonable and doable idea. Implementing a Latent Regression loss or a Mode Seeking loss alongside AdaIN-based noise injection into the existing CUT ResNet generator would effectively yield a "Stochastic CUT" model, multiplying our effective dataset size through randomized covers.

## Probabilistic Alpha Blending for CUT Augmentation
**Date:** 2026-06-05
**Status:** Proposed

### Motivation
When applying CUT-generated blanket covers to uncovered IR images, the generator produces a cover with a fixed "thickness" or opacity. In reality, different blankets have different thermal transmittances. A thinner blanket will let more of the original body's heat signature (high-frequency details) pass through.

### Proposed Idea
Merge the CUT-generated image with the original uncovered image using random weights (Alpha Blending):
`Augmented = alpha * CUT_Image + (1 - alpha) * Original_Image`
where `alpha` is sampled from a distribution, e.g., `Uniform(0.5, 1.0)`.

### Analysis
This makes perfect physical sense in the thermal (IR) domain because infrared radiation diffusion through a medium can be approximately modeled linearly in pixel space (temperature intensities). By varying `alpha`, we effectively simulate blankets of different thicknesses and materials, acting as an extremely cheap and effective way to introduce stochasticity into the deterministic CUT generator.

However, it may be problematic if `alpha` is too low (e.g., `< 0.4`), because the network might just learn to exploit the clearly visible underlying body features, destroying the regularization benefit of the cover. Therefore, `alpha` should be constrained to values where the cover remains the dominant visual feature.
