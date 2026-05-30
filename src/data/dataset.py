"""
Dataset implementations for the Simultaneously-collected Multimodal Lying Pose (SLP) dataset.
Supports multiple modalities (RGB, IR, Depth) and covers (uncover, cover1, cover2).
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union, Iterable, cast, Sized
import scipy.io as sio
from torch.utils.data import Dataset, default_collate
from pathlib import Path
from PIL import Image
from .augmentations import DataAugmenter
import torchvision.transforms.v2 as v2


def collate_skip_none(
    batch: List[Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Custom collate_fn that drops samples missing a target heatmap.
    
    Required because unannotated samples (covered subjects without labels)
    return target=None, which PyTorch's default collate cannot handle.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Collated batch, or None if the batch is empty after filtering.
    """
    clean_batch = [
        item for item in batch if item is not None and item.get("target") is not None
    ]
    if not clean_batch:
        return None
    return cast(Dict[str, Any], default_collate(clean_batch))


class VIPCupDataset(Dataset):
    """
    Simultaneously-collected Multimodal Lying Pose (SLP) dataset for IEEE VIP Cup 2021.
    Handles subject splits, modalities, covers, and Gaussian heatmap generation.
    """

    root: Path
    split: str
    subjects: Iterable[int]
    modalities: List[str]
    covers: List[str]
    transform: Optional[Any]
    augmenter: Optional[DataAugmenter]
    image_size: Tuple[int, int]
    heatmap_size: Tuple[int, int]
    sigma: float
    in_channels: int
    return_joints: bool
    samples: List[Dict[str, Any]]

    def __init__(
        self,
        root: Union[str, Path],
        subjects: Iterable[int] = range(1, 31),
        modalities: List[str] = ["RGB", "IR"],
        covers: Optional[List[str]] = ["uncover"],
        split: str = "train",
        transform: Optional[Any] = None,
        augmenter: Optional[DataAugmenter] = None,
        image_size: Tuple[int, int] = (256, 256),
        in_channels: int = 1,
        return_joints: bool = True,
    ) -> None:
        """
        Initializes the VIPCupDataset.

        Args:
            root: Root directory of the SLP dataset.
            subjects: Iterable of subject IDs to include.
            modalities: List of modalities to load (e.g., ['RGB', 'IR']).
            covers: List of cover types to include (e.g., ['uncover', 'cover1']).
            split: Dataset split ('train' or 'valid').
            transform: torchvision transforms to apply to images.
            augmenter: Custom DataAugmenter for geometric and domain augmentations.
            image_size: Target image resolution (Height, Width).
            in_channels: Number of input channels (1 for grayscale IR, 3 for RGB).
            return_joints: If True, returns joint coordinates and heatmaps.
        """
        self.root = Path(root)
        self.split = split  # "train" or "valid"
        self.subjects = subjects
        self.modalities = modalities
        if covers is None:
            self.covers = ["uncover", "cover1", "cover2"]
        else:
            self.covers = covers
        self.transform = transform
        self.augmenter = augmenter
        if self.augmenter is not None and hasattr(self.augmenter, "dataset_root"):
            setattr(self.augmenter, "dataset_root", str(self.root))
            # Re-initialize reference bank if it's AdvancedCoverAugmenter
            if hasattr(self.augmenter, "advanced_cover"):
                adv_cover = getattr(self.augmenter, "advanced_cover")
                setattr(adv_cover, "dataset_root", self.root)
                if hasattr(adv_cover, "_load_reference_bank"):
                    getattr(adv_cover, "_load_reference_bank")()
        self.image_size = image_size
        self.heatmap_size = (64, 64)  # HRNet output size
        self.sigma = 2.0
        self.in_channels = in_channels
        self.return_joints = return_joints

        self.samples = self._prepare_samples()
        if self.subjects and not self.samples:
            raise ValueError(
                f"No samples found for split='{self.split}', subjects={self.subjects}, covers={self.covers}. "
                f"Check your data path: {self.root}"
            )

    def set_sigma(self, sigma: float) -> None:
        """
        Updates the Gaussian sigma for heatmap generation.
        Used for dynamic sigma curriculum during training.

        Args:
            sigma: New standard deviation for Gaussian peaks.
        """
        self.sigma = sigma

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """
        Scans the filesystem and groups images with their corresponding annotations.

        Returns:
            List of sample dictionaries containing paths and joint data.
        """
        samples: List[Dict[str, Any]] = []
        # Determine which top-level split folder to search first
        split_order = (
            ["valid", "train", ""] if self.split == "valid" else ["train", "valid", ""]
        )
        for subject_id in self.subjects:
            subject_names = [f"{subject_id:05d}", f"Subject_{subject_id:02d}"]
            subj_dir: Optional[Path] = None
            for sn in subject_names:
                for nesting in split_order + [f"{s}/{s}" for s in split_order if s]:
                    potential = self.root / nesting / sn
                    if potential.exists() and any(potential.iterdir()):
                        subj_dir = potential
                        break
                if subj_dir:
                    break

            if not subj_dir:
                continue

            # Load annotations if available
            annotations: Dict[str, Optional[np.ndarray]] = {}
            for mod in self.modalities:
                mat_path = subj_dir / f"joints_gt_{mod}.mat"
                if mat_path.exists():
                    mat_data = sio.loadmat(str(mat_path))
                    # joints_gt shape is usually (3, 14, N) -> (coords, joints, images)
                    # coords: [x, y, occluded]
                    ann = cast(np.ndarray, mat_data["joints_gt"]).copy()
                    # Apply -1 shift to x and y
                    ann[:2, :, :] -= 1
                    annotations[mod] = ann
                else:
                    annotations[mod] = None

            # Find images for each modality and cover, then group them
            for cover in self.covers:
                # Collect paths for each modality
                mod_paths: Dict[str, List[Path]] = {}
                for mod in self.modalities:
                    mod_dir = subj_dir / mod / cover
                    if not mod_dir.exists():
                        # Fallback for structured root-level modalities or other layouts
                        fallbacks = [
                            subj_dir / mod / cover,
                            subj_dir / "train" / mod / cover,
                            subj_dir / mod,
                        ]
                        for fb in fallbacks:
                            if fb.exists():
                                mod_dir = fb
                                break

                    if mod_dir.exists():
                        img_files = sorted(
                            list(mod_dir.glob("*.jpg")) + list(mod_dir.glob("*.png"))
                        )
                        mod_paths[mod] = img_files

                if not mod_paths:
                    continue

                # Determine number of scenes (assume aligned if multiple modalities)
                num_scenes = min(len(paths) for paths in mod_paths.values())

                for i in range(num_scenes):
                    sample: Dict[str, Any] = {
                        "subject": subject_id,
                        "cover": cover,
                        "index": i,
                        "image_paths": {
                            mod: paths[i] for mod, paths in mod_paths.items()
                        },
                        "joints": {},
                    }

                    # Store joints for each modality if available
                    for mod in self.modalities:
                        ann_mod = annotations.get(mod)
                        if ann_mod is not None and i < ann_mod.shape[2]:
                            sample["joints"][mod] = ann_mod[:, :, i]
                        else:
                            sample["joints"][mod] = None

                    samples.append(sample)
        return samples

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset.
        Handles image loading, augmentation, resizing, and heatmap generation.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary containing image, joints, heatmaps, and metadata.
        """
        sample = self.samples[idx]

        # Default to IR for training/evaluation as requested
        # If IR is not available for some reason (shouldn't happen with min()), fallback to first available
        target_mod = (
            "IR"
            if "IR" in sample["image_paths"]
            else list(sample["image_paths"].keys())[0]
        )
        image_path = sample["image_paths"][target_mod]

        # Load and convert to 1-channel (L) or 3-channel (RGB) for IR
        if target_mod == "IR":
            if self.in_channels == 3:
                image: Union[Image.Image, torch.Tensor] = Image.open(
                    image_path
                ).convert("RGB")
            else:
                image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

        joints = sample["joints"].get(target_mod) if self.return_joints else None

        # Apply data augmentation if provided (affects both image and joints)
        image_source: Optional[Union[Image.Image, torch.Tensor]] = None
        if self.augmenter and self.split == "train":
            # For UDA, we want both the occluded (target) and clean (source) versions
            image, image_source, joints = self.augmenter(
                image, joints, is_ir=(target_mod == "IR"), return_pair=True
            )
        elif self.augmenter:
            image, joints = self.augmenter(image, joints, is_ir=(target_mod == "IR"))

        # Resize to standard size if not already handled by augmentation
        if hasattr(image, "size") and getattr(image, "size") != self.image_size:
            image = cast(Image.Image, image).resize(self.image_size)
            if image_source and hasattr(image_source, "size"):
                image_source = cast(Image.Image, image_source).resize(self.image_size)

        if joints is not None:
            # Need to scale joints if image was resized
            with Image.open(image_path) as img_orig:
                orig_w, orig_h = img_orig.size
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h

            scaled_joints = joints.copy()
            scaled_joints[0] *= scale_x
            scaled_joints[1] *= scale_y
            joints_tensor = torch.from_numpy(scaled_joints).float()
        else:
            joints_tensor = None

        # Convert to tensor if not already (augmenter might return tensors or PIL)
        if not torch.is_tensor(image):
            image = v2.functional.to_image(image).float() / 255.0

        if image_source is not None and not torch.is_tensor(image_source):
            image_source = v2.functional.to_image(image_source).float() / 255.0

        if self.transform:
            image = self.transform(image)
            if image_source:
                image_source = self.transform(image_source)

        target_heatmaps: Optional[torch.Tensor] = None
        if joints_tensor is not None:
            target_heatmaps = self._generate_heatmaps(joints_tensor)

        res: Dict[str, Any] = {
            "image": image,
            "joints": joints_tensor,
            "target": target_heatmaps,
            "subject": sample["subject"],
            "modality": target_mod,
            "cover": sample["cover"],
            "image_paths": {k: str(v) for k, v in sample["image_paths"].items()},
        }
        if image_source is not None:
            res["image_source"] = image_source
        return res

    def _generate_heatmaps(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D Gaussian heatmaps for each joint.
        joints: tensor of shape (3, 14) -> (x, y, visibility)
        """
        num_joints = joints.shape[1]
        heatmaps = np.zeros(
            (num_joints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32
        )

        # Scale joints to heatmap size
        scale_x = self.heatmap_size[0] / self.image_size[0]
        scale_y = self.heatmap_size[1] / self.image_size[1]

        for i in range(num_joints):
            # Dataset README: if_occluded == 0 means VISIBLE, != 0 means occluded.
            # In many tasks (like VIP Cup), we want to predict the pose even if occluded (under blanket).
            # We skip only if the joint is completely missing/unannotated (coords at 0,0).
            if joints[2, i] > 1:  # Type 2 is usually 'not annotated' or 'out of view'
                continue

            if joints[0, i] == 0 and joints[1, i] == 0:
                continue

            mu_x = int(joints[0, i] * scale_x + 0.5)
            mu_y = int(joints[1, i] * scale_y + 0.5)

            # Check bounds
            if (
                mu_x < 0
                or mu_y < 0
                or mu_x >= self.heatmap_size[0]
                or mu_y >= self.heatmap_size[1]
            ):
                continue

            # Generate gaussian
            size = int(6 * self.sigma + 1)
            if size % 2 == 0:
                size += 1
            x = np.arange(0, size, 1, dtype=float)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

            # Heatmap corners
            ul = [int(mu_x - x0), int(mu_y - y0)]
            br = [int(mu_x + x0 + 1), int(mu_y + y0 + 1)]

            # Range of gaussian
            g_x = [max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]]
            g_y = [max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]]

            # Range of heatmap
            img_x = [max(0, ul[0]), min(br[0], self.heatmap_size[0])]
            img_y = [max(0, ul[1]), min(br[1], self.heatmap_size[1])]

            heatmaps[i, img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                g_y[0] : g_y[1], g_x[0] : g_x[1]
            ]

        return torch.from_numpy(heatmaps)


class PairedDataset(Dataset):
    """
    Wraps two datasets and returns pairs.
    Useful for CycleGAN/unpaired translation.
    """

    ds_a: Dataset
    ds_b: Dataset

    def __init__(self, ds_a: Dataset, ds_b: Dataset) -> None:
        self.ds_a = ds_a
        self.ds_b = ds_b

    def __len__(self) -> int:
        return max(len(cast(Sized, self.ds_a)), len(cast(Sized, self.ds_b)))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample randomly from both to handle size mismatch
        idx_a = torch.randint(0, len(cast(Sized, self.ds_a)), (1,)).item()
        idx_b = torch.randint(0, len(cast(Sized, self.ds_b)), (1,)).item()

        sample_a = self.ds_a[idx_a]
        sample_b = self.ds_b[idx_b]

        # Return just the images for CycleGAN
        img_a = sample_a["image"]
        img_b = sample_b["image"]

        # Generator expects normalized [-1, 1] for Tanh output
        if isinstance(img_a, torch.Tensor) and img_a.max() <= 1.0:
            img_a = (img_a * 2.0) - 1.0
        if isinstance(img_b, torch.Tensor) and img_b.max() <= 1.0:
            img_b = (img_b * 2.0) - 1.0

        return img_a, img_b
