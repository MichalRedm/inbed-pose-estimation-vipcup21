import random
import numpy as np
import torch
import inspect
from PIL import Image
from torchvision import tv_tensors
from typing import Any, Optional, Union, Dict, List, Tuple, cast

from .geometric import HorizontalFlipAugmentation, AffineAugmentation
from .intensity import ThermalIntensityJitter, IRSensorNoise
from .thermal import ThermalDiffusionAugmenter, AdvancedCoverAugmenter
from .occlusion import CutoutAugmentation
from .domain import CycleGANAugmentation, CUTAugmentation


class DataAugmenter:
    """
    Optimized Data Augmentation using torchvision.transforms.v2.
    """

    config: Dict[str, Any]
    enabled: bool
    is_training: bool
    dataset_root: str
    flip: HorizontalFlipAugmentation
    affine: AffineAugmentation
    intensity_jitter: ThermalIntensityJitter
    sensor_noise: IRSensorNoise
    thermal_augmenter: ThermalDiffusionAugmenter
    advanced_cover: AdvancedCoverAugmenter
    cyclegan: CycleGANAugmentation
    cut: CUTAugmentation
    cutout: CutoutAugmentation
    exclusive_occlusion: bool

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
        dataset_root: Optional[str] = None,
    ) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.is_training = is_training
        self.dataset_root = dataset_root or str(
            self.config.get("dataset_root", "data/SLP")
        )

        # Components
        self.flip = HorizontalFlipAugmentation(
            probability=float(self.config.get("flip_prob", 0.5))
        )

        rot_range = cast(List[float], self.config.get("rotation_range", [-30.0, 30.0]))
        scale_range = cast(List[float], self.config.get("scaling_range", [0.8, 1.2]))
        translate = cast(Optional[List[float]], self.config.get("translation", None))
        self.affine = AffineAugmentation(rot_range, scale_range, translate)

        self.intensity_jitter = ThermalIntensityJitter(
            probability=float(self.config.get("intensity_jitter_prob", 0.5)),
            brightness_range=cast(
                List[float], self.config.get("intensity_jitter_range", [0.55, 1.15])
            ),
            contrast_range=cast(
                List[float], self.config.get("contrast_jitter_range", [0.5, 1.15])
            ),
        )

        self.sensor_noise = IRSensorNoise(
            probability=float(self.config.get("sensor_noise_prob", 0.4)),
            sigma_range=cast(
                List[float], self.config.get("sensor_noise_sigma", [5.0, 12.0])
            ),
        )

        # Occlusion methods
        self.thermal_augmenter = ThermalDiffusionAugmenter(
            probability=float(self.config.get("occlusion_prob", 0.5)),
            is_training=self.is_training,
        )

        self.advanced_cover = AdvancedCoverAugmenter(
            dataset_root=self.dataset_root,
            probability=float(self.config.get("advanced_cover_prob", 0.0)),
            bank_size=int(self.config.get("advanced_cover_bank_size", 100)),
        )

        self.cyclegan = CycleGANAugmentation(
            probability=float(self.config.get("cyclegan_prob", 0.0)),
            checkpoint_path=str(
                self.config.get("cyclegan_path", "models/cyclegan_gen_A2B.pth")
            ),
        )

        self.cut = CUTAugmentation(
            probability=float(self.config.get("cut_prob", 0.0)),
            checkpoint_path=str(self.config.get("cut_path", "models/cut_gen.pth")),
        )

        self.cutout = CutoutAugmentation(
            probability=float(self.config.get("cutout_prob", 0.5)),
            size_ratio=float(self.config.get("cutout_size_ratio", 0.35)),
        )

        self.exclusive_occlusion = bool(self.config.get("exclusive_occlusion", False))

    def __call__(
        self,
        image: Union[Image.Image, torch.Tensor],
        joints: Optional[np.ndarray],
        is_ir: bool = False,
        return_pair: bool = False,
    ) -> Any:
        if not self.enabled:
            if return_pair:
                return image, image, joints
            return image, joints

        # 1. Prepare Keypoints
        kpts: Optional[tv_tensors.Keypoints] = None
        vis: Optional[torch.Tensor] = None

        if joints is not None:
            if torch.is_tensor(joints):
                # Assume (3, 14) or (N, 2)
                if joints.shape[0] == 3:
                    coords = joints[:2, :].T.unsqueeze(0)
                    vis = joints[2, :]
                else:
                    coords = joints.unsqueeze(0)
                    vis = torch.ones(joints.shape[0], device=joints.device)
            else:
                coords = torch.from_numpy(joints[:2, :].T).float().unsqueeze(0)
                vis = torch.from_numpy(joints[2, :]).float()

            if hasattr(image, "width"):
                img_pil = cast(Image.Image, image)
                w, h = img_pil.width, img_pil.height
            else:
                img_tensor = cast(torch.Tensor, image)
                w, h = img_tensor.shape[-1], img_tensor.shape[-2]

            kpts = tv_tensors.Keypoints(coords, canvas_size=(h, w))

        # 2. Sequential Application
        if self.is_training:
            image, kpts = self.flip(image, kpts)

        if kpts is not None:
            image, kpts = self.affine(image, kpts)
        else:
            image, _ = self.affine(image)

        image = self.intensity_jitter(image)
        image = self.sensor_noise(image)

        source_image = image
        if return_pair:
            source_image = (
                image.clone()
                if torch.is_tensor(image)
                else cast(Image.Image, image).copy()
            )

        # Occlusion block
        if self.exclusive_occlusion:
            candidates = []
            if random.random() < self.cyclegan.probability:
                candidates.append("cyclegan")
            if random.random() < self.cut.probability:
                candidates.append("cut")
            if random.random() < self.advanced_cover.probability:
                candidates.append("advanced_cover")
            if random.random() < self.thermal_augmenter.probability:
                candidates.append("thermal")

            if candidates:
                choice = random.choice(candidates)
                if choice == "cyclegan":
                    image = self.cyclegan(image, force_apply=True)
                elif choice == "cut":
                    image = self.cut(image, force_apply=True)
                elif choice == "advanced_cover":
                    image = self.advanced_cover(
                        image,
                        joints=kpts,
                        is_ir=is_ir,
                        force_apply=True,
                        fda_prob=float(self.config.get("fda_prob", 0.5)),
                        hist_match_prob=float(self.config.get("hist_match_prob", 0.5)),
                        fda_beta=float(self.config.get("fda_beta", 0.01)),
                    )
                elif choice == "thermal":
                    image = self.thermal_augmenter(
                        image, joints=kpts, is_ir=is_ir, force_apply=True
                    )
        else:
            image = self.cyclegan(image)
            image = self.cut(image)
            image = self.advanced_cover(
                image,
                joints=kpts,
                is_ir=is_ir,
                fda_prob=float(self.config.get("fda_prob", 0.5)),
                hist_match_prob=float(self.config.get("hist_match_prob", 0.5)),
                fda_beta=float(self.config.get("fda_beta", 0.01)),
            )
            image = self.thermal_augmenter(image, joints=kpts, is_ir=is_ir)

        image = self.cutout(image)

        # 3. Final Assembly
        final_joints: Optional[np.ndarray] = None
        if kpts is not None:
            num_kpts = kpts.shape[1]
            final_coords = kpts.view(num_kpts, 2).T  # (2, num_kpts)
            if vis is not None:
                final_joints = torch.cat(
                    [final_coords, vis.unsqueeze(0)], dim=0
                ).numpy()
            else:
                final_joints = final_coords.numpy()

        if return_pair:
            return image, source_image, final_joints
        return image, final_joints


def get_available_augmentations() -> List[Dict[str, Any]]:
    """
    Returns metadata for all discoverable augmentation classes in the sub-modules.
    """
    augmentations: List[Dict[str, Any]] = []
    # We can hardcode them or use inspect on the imported modules
    import src.data.augmentations.geometric as geo
    import src.data.augmentations.intensity as intense
    import src.data.augmentations.thermal as therm
    import src.data.augmentations.occlusion as occ
    import src.data.augmentations.domain as dom

    for module in [geo, intense, therm, occ, dom]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "METADATA"):
                augmentations.append(getattr(obj, "METADATA"))

    return sorted(augmentations, key=lambda x: int(x.get("order", 99)))


def apply_custom_augmentations(
    image: Image.Image,
    joints: Optional[np.ndarray],
    aug_list: List[Dict[str, Any]],
    is_ir: bool = False,
    dataset_root: Optional[str] = None,
) -> Tuple[Union[Image.Image, torch.Tensor], Optional[np.ndarray]]:
    """
    Dynamically applies a list of augmentations.
    """
    # Map IDs to classes
    import src.data.augmentations.geometric as geo
    import src.data.augmentations.intensity as intense
    import src.data.augmentations.thermal as therm
    import src.data.augmentations.occlusion as occ
    import src.data.augmentations.domain as dom

    all_classes: Dict[str, Any] = {}
    for module in [geo, intense, therm, occ, dom]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "METADATA"):
                all_classes[getattr(obj, "METADATA")["id"]] = obj

    # Prepare kpts
    kpts: Optional[tv_tensors.Keypoints] = None
    vis: Optional[torch.Tensor] = None
    if joints is not None:
        coords = torch.from_numpy(joints[:2, :].T).float().unsqueeze(0)
        vis = torch.from_numpy(joints[2, :]).float()
        w, h = image.width, image.height
        kpts = tv_tensors.Keypoints(coords, canvas_size=(h, w))

    current_image: Union[Image.Image, torch.Tensor] = image

    for aug_cfg in aug_list:
        aug_id = str(aug_cfg.get("id"))
        params = dict(aug_cfg.get("params", {}))
        if aug_id in all_classes:
            aug_cls = all_classes[aug_id]
            if "probability" not in params:
                params["probability"] = 1.0

            # Special case for AdvancedCoverAugmenter
            if aug_id == "advanced_cover":
                inst = aug_cls(dataset_root=dataset_root or "data/SLP")
            else:
                inst = aug_cls()

            if aug_id in ["flip", "affine"]:
                current_image, kpts = inst(current_image, kpts, **params)
            elif aug_id in ["thermal_diffusion", "advanced_cover"]:
                current_image = inst(current_image, joints=kpts, is_ir=is_ir, **params)
            else:
                current_image = inst(current_image, **params)

    # Final assembly
    final_joints: Optional[np.ndarray] = None
    if kpts is not None:
        num_kpts = kpts.shape[1]
        final_coords = kpts.view(num_kpts, 2).T
        if vis is not None:
            final_joints = torch.cat([final_coords, vis.unsqueeze(0)], dim=0).numpy()
        else:
            final_joints = final_coords.numpy()

    return current_image, final_joints
