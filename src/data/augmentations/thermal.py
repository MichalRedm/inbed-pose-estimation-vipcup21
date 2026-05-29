import random
import numpy as np
import torch
import torch.fft
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms.v2 as v2
from typing import Union, Optional
from pathlib import Path


def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Matches the histogram of the source image to the reference image.
    source, reference: np.ndarray (H, W) or (H, W, C) in [0, 255]
    """
    if source.ndim == 2:
        return _match_cumulative_distribution(source, reference)
    else:
        res = np.zeros_like(source)
        # Handle cases where reference might have different number of channels
        ref_channels = reference.shape[2] if reference.ndim == 3 else 1
        for i in range(source.shape[2]):
            ref_chan = (
                reference[:, :, i % ref_channels] if reference.ndim == 3 else reference
            )
            res[:, :, i] = _match_cumulative_distribution(source[:, :, i], ref_chan)
        return res


def _match_cumulative_distribution(
    source: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    src_values, src_indices, src_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(reference, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference.size

    interp_values = np.interp(src_cdf, ref_cdf, ref_values)
    return interp_values[src_indices].reshape(source.shape).astype(source.dtype)


def fourier_domain_adaptation(
    src_img: torch.Tensor, trg_img: torch.Tensor, beta: float = 0.01
) -> torch.Tensor:
    """
    Fourier Domain Adaptation (FDA)
    src_img, trg_img: torch.Tensor [C, H, W]
    beta: boundary for low-frequency swap
    """
    # Get FFT
    fft_src = torch.fft.fftn(src_img, dim=(-2, -1))
    fft_trg = torch.fft.fftn(trg_img, dim=(-2, -1))

    # Shift to center
    fft_src_shifted = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_trg_shifted = torch.fft.fftshift(fft_trg, dim=(-2, -1))

    # Get mask for low frequencies
    _, H, W = src_img.shape
    b = int(np.floor(min(H, W) * beta))
    if b < 1:
        b = 1

    cy, cx = H // 2, W // 2

    # Original FDA paper swaps the amplitude and keeps the phase of src.
    amp_src = torch.abs(fft_src_shifted)
    pha_src = torch.angle(fft_src_shifted)
    amp_trg = torch.abs(fft_trg_shifted)

    # Apply swap to amplitude
    # We clone to avoid modifying original tensors if they are reused
    amp_src_mutated = amp_src.clone()
    amp_src_mutated[:, cy - b : cy + b, cx - b : cx + b] = amp_trg[
        :, cy - b : cy + b, cx - b : cx + b
    ]

    # Reconstruct
    fft_src_mutated = amp_src_mutated * torch.exp(1j * pha_src)

    # Inverse shift and inverse FFT
    fft_src_mutated = torch.fft.ifftshift(fft_src_mutated, dim=(-2, -1))
    src_in_trg = torch.fft.ifftn(fft_src_mutated, dim=(-2, -1)).real

    return src_in_trg


class ThermalDiffusionAugmenter:
    """
    Simulates the effect of a blanket on IR images by diffusing and dampening the heat signature.
    """

    METADATA = {
        "id": "thermal_diffusion",
        "name": "Thermal Diffusion (Blanket)",
        "order": 4,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "base_y_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "damp_factor": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.55},
            "blur_radius": {"type": "float", "min": 0.0, "max": 20.0, "default": 7.0},
        },
    }

    def __init__(self, probability: float = 0.5, is_training: bool = True):
        self.probability = probability
        self.is_training = is_training

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[torch.Tensor] = None,
        is_ir: bool = True,
        **kwargs,
    ) -> Union[Image.Image, torch.Tensor]:
        force = kwargs.get("force_apply", False)
        prob = kwargs.get("probability", self.probability)
        if not is_ir or (not force and random.random() > prob):
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        original_mode = img_pil.mode
        if original_mode != "L":
            img_pil = img_pil.convert("L")

        w, h = img_pil.size

        # 1. Determine blanket Y start position (base_y)
        head_y = None
        shoulders_y = []
        hips_y = []
        knees_y = []

        if joints is not None:
            if torch.is_tensor(joints):
                if len(joints.shape) == 3:  # (1, N, 2)
                    j_np = joints[0].cpu().numpy()
                elif len(joints.shape) == 2:  # (N, 2) or (3, N)
                    if joints.shape[0] == 3:
                        j_np = joints[:2, :].T.cpu().numpy()
                    else:
                        j_np = joints.cpu().numpy()
                else:
                    j_np = np.array(joints)
            else:
                j_np = np.array(joints)

            if len(j_np.shape) == 2 and j_np.shape[0] >= 14:
                if j_np[13, 0] > 0 or j_np[13, 1] > 0:
                    head_y = j_np[13, 1]
                for s_idx in [8, 9]:
                    if j_np[s_idx, 0] > 0 or j_np[s_idx, 1] > 0:
                        shoulders_y.append(j_np[s_idx, 1])
                for h_idx in [2, 3]:
                    if j_np[h_idx, 0] > 0 or j_np[h_idx, 1] > 0:
                        hips_y.append(j_np[h_idx, 1])
                for k_idx in [1, 4]:
                    if j_np[k_idx, 0] > 0 or j_np[k_idx, 1] > 0:
                        knees_y.append(j_np[k_idx, 1])

        min_allowed_y = int(h * 0.15)
        if head_y is not None:
            min_allowed_y = max(min_allowed_y, int(head_y + 15))

        if "base_y_ratio" in kwargs:
            base_y = int(kwargs["base_y_ratio"] * h)
            base_y = max(min_allowed_y, base_y)
        else:
            coverage_choices = []
            if shoulders_y:
                coverage_choices.append(min(shoulders_y))
            if hips_y:
                coverage_choices.append(min(hips_y))
            if knees_y:
                coverage_choices.append(min(knees_y))
            if coverage_choices:
                base_y = random.choice(coverage_choices)
                base_y += random.randint(-15, 15)
            else:
                base_y = random.randint(int(h * 0.25), int(h * 0.7))
            base_y = max(min_allowed_y, min(base_y, h - 20))

        base_y = int(base_y)

        # 2. Create wavy mask
        mask = Image.new("L", img_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        wavy_points = []
        num_points = 20
        freq = random.uniform(1.5, 3.5)
        amp = random.uniform(4, 15)
        phase = random.uniform(0, 2 * np.pi)

        for i in range(num_points + 1):
            x = int(i * w / num_points)
            y = int(base_y + amp * np.sin(freq * (x / w) * 2 * np.pi + phase))
            y = max(0, min(h - 1, y))
            wavy_points.append((x, y))

        polygon_points = list(wavy_points)
        polygon_points.append((w, h))
        polygon_points.append((0, h))
        draw.polygon(polygon_points, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 7)))

        # 3. Get image as numpy and estimate ambient
        img_np = np.array(img_pil).astype(np.float32)
        ambient_est = np.percentile(img_np, 15)

        # 4. Generate drape and wrinkle masks
        drape_mask = Image.new("L", img_pil.size, 255)
        drape_draw = ImageDraw.Draw(drape_mask)
        for _ in range(random.randint(3, 6)):
            fx1, fy1 = random.randint(-40, w + 40), random.randint(base_y - 40, h + 40)
            fx2, fy2 = random.randint(-40, w + 40), fy1 + random.randint(80, 250)
            cx, cy = (
                random.randint(min(fx1, fx2) - 50, max(fx1, fx2) + 50),
                (fy1 + fy2) // 2 + random.randint(-20, 20),
            )
            pts = [
                (
                    int((1 - t) ** 2 * fx1 + 2 * (1 - t) * t * cx + t**2 * fx2),
                    int((1 - t) ** 2 * fy1 + 2 * (1 - t) * t * cy + t**2 * fy2),
                )
                for t in np.linspace(0, 1, 15)
            ]
            drape_draw.line(
                pts,
                fill=random.randint(40, 120),
                width=random.randint(25, 50),
                joint="round",
            )
        drape_np = (
            np.array(
                drape_mask.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(10.0, 18.0))
                )
            ).astype(np.float32)
            / 255.0
        )

        wrinkle_mask = Image.new("L", img_pil.size, 255)
        wrinkle_draw = ImageDraw.Draw(wrinkle_mask)
        for _ in range(random.randint(4, 8)):
            fx1, fy1 = random.randint(-40, w + 40), random.randint(base_y - 20, h + 40)
            fx2, fy2 = fx1 + random.randint(-60, 60), fy1 + random.randint(40, 180)
            cx, cy = (
                (fx1 + fx2) // 2 + random.randint(-15, 15),
                (fy1 + fy2) // 2 + random.randint(-10, 10),
            )
            pts = [
                (
                    int((1 - t) ** 2 * fx1 + 2 * (1 - t) * t * cx + t**2 * fx2),
                    int((1 - t) ** 2 * fy1 + 2 * (1 - t) * t * cy + t**2 * fy2),
                )
                for t in np.linspace(0, 1, 10)
            ]
            wrinkle_draw.line(
                pts,
                fill=random.randint(80, 160),
                width=random.randint(3, 7),
                joint="round",
            )
        wrinkle_np = (
            np.array(
                wrinkle_mask.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.5))
                )
            ).astype(np.float32)
            / 255.0
        )

        combined_drape_np = drape_np * wrinkle_np
        blanket_base = (
            ambient_est
            + (combined_drape_np - 0.85) * 12.0
            + np.random.normal(0, 1.5, img_np.shape).astype(np.float32)
        )

        # 6. Simulate body heat
        body_heat = np.maximum(img_np - ambient_est, 0.0)
        body_heat_norm = body_heat / (np.max(body_heat) + 1e-5)
        body_heat_boosted = np.power(body_heat_norm, 1.3) * np.max(body_heat)
        heat_pil = Image.fromarray(np.clip(body_heat_boosted, 0, 255).astype(np.uint8))
        bloom_np = np.array(
            heat_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(8.0, 16.0)))
        ).astype(np.float32)
        contact_np = np.array(
            heat_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 5.0)))
        ).astype(np.float32)
        mixed_heat_np = (
            bloom_np * (1.0 - combined_drape_np) + contact_np * combined_drape_np
        )

        damp_factor = kwargs.get("damp_factor", random.uniform(0.18, 0.42))
        dampened_np = blanket_base + mixed_heat_np * damp_factor * combined_drape_np

        # 8. Shadow
        shadow_mask = Image.new("L", img_pil.size, 0)
        ImageDraw.Draw(shadow_mask).line(
            wavy_points, fill=255, width=random.randint(4, 8), joint="round"
        )
        shadow_np = (
            np.array(
                shadow_mask.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(2, 5))
                )
            ).astype(np.float32)
            / 255.0
        )
        dampened_np = dampened_np * (1.0 - shadow_np * random.uniform(0.03, 0.08))

        dampened = Image.fromarray(np.clip(dampened_np, 0, 255).astype(np.uint8))
        final_image = Image.composite(dampened, img_pil, mask)

        if original_mode != "L":
            final_image = final_image.convert(original_mode)
        if is_tensor:
            return v2.functional.to_image(final_image).to(device)
        return final_image


class AdvancedCoverAugmenter:
    """
    Enhanced Synthetic Cover Augmentation using Histogram Matching and FDA.
    Uses a dynamic reference bank of real covered images from the training set.
    """

    METADATA = {
        "id": "advanced_cover",
        "name": "Advanced Synthetic Cover",
        "order": 4,
        "params": {
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "fda_prob": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "hist_match_prob": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
            },
            "fda_beta": {"type": "float", "min": 0.001, "max": 0.1, "default": 0.01},
        },
    }

    def __init__(
        self, dataset_root: str, probability: float = 0.5, bank_size: int = 100
    ):
        self.probability = probability
        self.bank_size = bank_size
        self.dataset_root = Path(dataset_root)
        self.reference_bank = []
        self._load_reference_bank()

    def _load_reference_bank(self):
        """Find real covered images in the training set (Subjects 1-80)."""
        covered_images = []

        # We look for 'cover1' and 'cover2' directories within training subject folders.
        # Training subjects are usually 1-80.
        # Structure can be: root/train/Subject_XX/IR/coverX/ or root/train/train/000XX/IR/coverX/

        # Search all possible training directories
        search_roots = [
            self.dataset_root / "train",
            self.dataset_root / "train" / "train",
            self.dataset_root,
        ]

        for root in search_roots:
            if not root.exists():
                continue
            for cover in ["cover1", "cover2"]:
                # Recursive glob to find images inside cover1/cover2 directories
                # This is more robust to different nesting levels
                covered_images.extend(list(root.rglob(f"**/IR/{cover}/*.png")))
                covered_images.extend(list(root.rglob(f"**/IR/{cover}/*.jpg")))

                # Also try without IR subfolder just in case
                if not covered_images:
                    covered_images.extend(list(root.rglob(f"**/{cover}/*.png")))
                    covered_images.extend(list(root.rglob(f"**/{cover}/*.jpg")))

        if not covered_images:
            # Fallback: search for any 'cover' folder
            covered_images.extend(list(self.dataset_root.rglob("**/cover*/*.png")))
            covered_images.extend(list(self.dataset_root.rglob("**/cover*/*.jpg")))

        # Filter for training subjects if possible (1-80)
        # In SLP, subjects 1-80 are training.
        # Paths usually contain Subject_XX or 000XX.
        final_images = []
        for img_path in covered_images:
            path_str = str(img_path)
            # Simple heuristic: if it contains a subject ID > 80, it might be validation/test
            # But usually they are in 'train' vs 'test' folders.
            # If we found them under a 'train' search root, they are likely safe.
            if "valid" in path_str.lower() or "test" in path_str.lower():
                continue
            final_images.append(img_path)

        if not final_images:
            print(
                f"Warning: AdvancedCoverAugmenter found no reference images in {self.dataset_root}"
            )
            return

        random.shuffle(final_images)
        self.reference_bank = final_images[: self.bank_size]
        print(
            f"AdvancedCoverAugmenter initialized with {len(self.reference_bank)} reference images."
        )

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[torch.Tensor] = None,
        is_ir: bool = True,
        **kwargs,
    ) -> Union[Image.Image, torch.Tensor]:
        force = kwargs.get("force_apply", False)
        prob = kwargs.get("probability", self.probability)
        if (
            not is_ir
            or not self.reference_bank
            or (not force and random.random() > prob)
        ):
            return image

        is_tensor = torch.is_tensor(image)
        if is_tensor:
            device = image.device
            img_pil = v2.functional.to_pil_image(image)
        else:
            img_pil = image

        original_mode = img_pil.mode
        if original_mode != "L":
            img_pil = img_pil.convert("L")

        w, h = img_pil.size

        # 1. Determine blanket Y start position (Logic from ThermalDiffusionAugmenter)
        head_y = None
        if joints is not None:
            if torch.is_tensor(joints):
                if len(joints.shape) == 3:
                    j_np = joints[0].cpu().numpy()
                elif len(joints.shape) == 2:
                    j_np = (
                        joints[:2, :].T.cpu().numpy()
                        if joints.shape[0] == 3
                        else joints.cpu().numpy()
                    )
                else:
                    j_np = np.array(joints)
            else:
                j_np = np.array(joints)
            if len(j_np.shape) == 2 and j_np.shape[0] >= 14:
                if j_np[13, 0] > 0 or j_np[13, 1] > 0:
                    head_y = j_np[13, 1]

        min_allowed_y = int(h * 0.15)
        if head_y is not None:
            min_allowed_y = max(min_allowed_y, int(head_y + 15))
        base_y = int(random.uniform(max(min_allowed_y, h * 0.25), h * 0.7))

        # 2. Create wavy mask
        mask_pil = Image.new("L", img_pil.size, 0)
        draw = ImageDraw.Draw(mask_pil)
        wavy_points = []
        num_points = 20
        freq, amp, phase = (
            random.uniform(1.5, 3.5),
            random.uniform(4, 15),
            random.uniform(0, 2 * np.pi),
        )
        for i in range(num_points + 1):
            x = int(i * w / num_points)
            y = int(base_y + amp * np.sin(freq * (x / w) * 2 * np.pi + phase))
            y = max(0, min(h - 1, y))
            wavy_points.append((x, y))
        polygon_points = list(wavy_points) + [(w, h), (0, h)]
        draw.polygon(polygon_points, fill=255)
        mask_pil = mask_pil.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(3, 7))
        )

        # 3. Select reference image and apply styles
        ref_path = random.choice(self.reference_bank)
        ref_pil = Image.open(ref_path).convert("L").resize((w, h))

        src_np = np.array(img_pil).astype(np.float32)
        ref_np = np.array(ref_pil).astype(np.float32)

        styled_np = src_np.copy()

        # Histogram Matching
        if random.random() < kwargs.get("hist_match_prob", 0.5):
            styled_np = histogram_matching(styled_np, ref_np)

        # FDA
        if random.random() < kwargs.get("fda_prob", 0.5):
            styled_t = torch.from_numpy(styled_np).unsqueeze(0) / 255.0
            ref_t = torch.from_numpy(ref_np).unsqueeze(0) / 255.0
            styled_t = fourier_domain_adaptation(
                styled_t, ref_t, beta=kwargs.get("fda_beta", 0.01)
            )
            styled_np = (styled_t.squeeze(0).numpy() * 255.0).clip(0, 255)

        styled_pil = Image.fromarray(styled_np.astype(np.uint8))

        # 4. Composite
        final_image = Image.composite(styled_pil, img_pil, mask_pil)

        if original_mode != "L":
            final_image = final_image.convert(original_mode)
        if is_tensor:
            return v2.functional.to_image(final_image).to(device)
        return final_image
