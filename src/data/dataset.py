import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, default_collate
from pathlib import Path
from PIL import Image


def collate_skip_none(batch):
    """
    Custom collate_fn that drops samples missing a target heatmap.
    Required because unannotated samples (covered subjects without labels)
    return target=None, which PyTorch's default collate cannot handle.
    """
    batch = [item for item in batch if item.get("target") is not None]
    if not batch:
        return None
    return default_collate(batch)


class VIPCupDataset(Dataset):
    """
    Simultaneously-collected Multimodal Lying Pose (SLP) dataset for IEEE VIP Cup 2021.
    """

    def __init__(
        self,
        root,
        subjects=range(1, 31),
        modalities=["RGB", "IR"],
        covers=["uncover"],
        split="train",
        transform=None,
        image_size=(256, 256),
    ):
        self.root = Path(root)
        self.split = split  # "train" or "valid"
        self.subjects = subjects
        self.modalities = modalities
        self.covers = covers
        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = (64, 64)  # HRNet output size
        self.sigma = 2.0

        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        # Determine which top-level split folder to search first
        split_order = (
            ["valid", "train", ""] if self.split == "valid" else ["train", "valid", ""]
        )
        for subject_id in self.subjects:
            subject_names = [f"{subject_id:05d}", f"Subject_{subject_id:02d}"]
            subj_dir = None
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
            annotations = {}
            for mod in self.modalities:
                mat_path = subj_dir / f"joints_gt_{mod}.mat"
                if mat_path.exists():
                    mat_data = sio.loadmat(mat_path)
                    # joints_gt shape is usually (3, 14, N) -> (coords, joints, images)
                    # coords: [x, y, occluded]
                    annotations[mod] = mat_data["joints_gt"]
                    # Apply -1 shift to x and y
                    annotations[mod][:2, :, :] -= 1
                else:
                    annotations[mod] = None

            # 3. Find images for each modality and cover
            for cover in self.covers:
                for mod in self.modalities:
                    # Generic search for modality subfolder
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

                    if not mod_dir.exists():
                        continue

                    img_files = sorted(
                        list(mod_dir.glob("*.jpg")) + list(mod_dir.glob("*.png"))
                    )
                    for i, img_path in enumerate(img_files):
                        sample = {
                            "image_path": img_path,
                            "subject": subject_id,
                            "modality": mod,
                            "cover": cover,
                            "index": i,
                        }
                        if (
                            annotations[mod] is not None
                            and i < annotations[mod].shape[2]
                        ):
                            sample["joints"] = annotations[mod][:, :, i]
                        else:
                            sample["joints"] = None

                        samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        # Resize to standard size
        image = image.resize(self.image_size)

        joints = sample["joints"]
        if joints is not None:
            # Need to scale joints if image was resized
            # Original sizes vary, but commonly 160x120 or similar for IR?
            # We'll need the original size for scaling.
            orig_w, orig_h = Image.open(sample["image_path"]).size
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h

            scaled_joints = joints.copy()
            scaled_joints[0] *= scale_x
            scaled_joints[1] *= scale_y
            joints = torch.from_numpy(scaled_joints).float()

        if self.transform:
            image = self.transform(image)
        else:
            # Default to tensor conversion
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        target_heatmaps = None
        if joints is not None:
            target_heatmaps = self._generate_heatmaps(joints)

        return {
            "image": image,
            "joints": joints,
            "target": target_heatmaps,
            "subject": sample["subject"],
            "modality": sample["modality"],
            "cover": sample["cover"],
        }

    def _generate_heatmaps(self, joints):
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
            # Only generate a heatmap for visible (non-occluded) joints.
            if joints[2, i] != 0:
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
            size = 6 * self.sigma + 1
            x = np.arange(0, size, 1, float)
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
