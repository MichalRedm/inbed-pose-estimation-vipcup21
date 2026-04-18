import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


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
        transform=None,
        image_size=(256, 256),
    ):
        self.root = Path(root)
        self.subjects = subjects
        self.modalities = modalities
        self.covers = covers
        self.transform = transform
        self.image_size = image_size

        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for subject_id in self.subjects:
            subj_str = f"{subject_id:05d}"
            subj_dir = (
                self.root / "train" / subj_str
            )  # Assuming 'train' subdir as per notebook

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

            # Iterate over positions/covers
            for cover in self.covers:
                # The structure might vary, checking RGB/uncover
                # Reference: subj_dir / MODALITY / COVER / *.jpg
                for mod in self.modalities:
                    img_dir = subj_dir / mod / cover
                    if not img_dir.exists():
                        continue

                    img_files = sorted(list(img_dir.glob("*.jpg")))
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

        return {
            "image": image,
            "joints": joints,
            "subject": sample["subject"],
            "modality": sample["modality"],
            "cover": sample["cover"],
        }
