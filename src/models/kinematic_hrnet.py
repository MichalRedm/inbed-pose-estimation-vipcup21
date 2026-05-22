import torch
import torch.nn as nn
import torch.nn.functional as F
from .hrnet import HRNet
from .layers import SoftArgmax2D
from .registry import register_model


class KinematicRefiner(nn.Module):
    """
    Kinematic Bone-Vector Decomposition refiner.
    Takes coordinates predicted by soft-argmax (B, 14, 2).
    Predicts:
      - Root offset (B, 2) to correct the root position (Thorax/Neck, joint 12).
      - Bone direction offsets (B, 13, 2).
      - Bone length scaling factors (B, 13).
    Reconstructs the full 14 joints recursively using differentiable PyTorch operations.
    """

    # LSP Kinematic Tree topological connections (parent, child, name)
    # Root: Neck/Thorax (index 12)
    BONES = [
        (12, 13, "Neck_Head"),
        (12, 8, "Neck_RShoulder"),
        (12, 9, "Neck_LShoulder"),
        (8, 7, "RShoulder_RElbow"),
        (7, 6, "RElbow_RWrist"),
        (9, 10, "LShoulder_LElbow"),
        (10, 11, "LElbow_LWrist"),
        (8, 2, "RShoulder_RHip"),
        (9, 3, "LShoulder_LHip"),
        (2, 1, "RHip_RKnee"),
        (1, 0, "RKnee_RAnkle"),
        (3, 4, "LHip_LKnee"),
        (4, 5, "LKnee_LAnkle"),
    ]

    def __init__(self, hidden_dim=128):
        super().__init__()

        # MLP to process flattened soft-argmax coordinates (14 joints * 2 coords = 28 features)
        self.mlp = nn.Sequential(
            nn.Linear(28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Regressors
        self.root_offset_head = nn.Linear(hidden_dim, 2)
        self.bone_direction_head = nn.Linear(hidden_dim, 13 * 2)
        self.bone_length_head = nn.Linear(hidden_dim, 13)

        # Buffer for baseline average bone lengths (updated at training start from dataset)
        self.register_buffer("avg_lengths", torch.ones(13) * 20.0)

        # Initialize heads with very small weights for initial identity mapping
        nn.init.zeros_(self.root_offset_head.weight)
        nn.init.zeros_(self.root_offset_head.bias)
        nn.init.zeros_(self.bone_direction_head.weight)
        nn.init.zeros_(self.bone_direction_head.bias)
        nn.init.zeros_(self.bone_length_head.weight)
        nn.init.zeros_(self.bone_length_head.bias)

    def forward(self, coords_initial):
        """
        Args:
            coords_initial: (B, 14, 2) initial keypoints from soft-argmax.
        Returns:
            coords_refined: (B, 14, 2) kinematically reconstructed pose.
        """
        B = coords_initial.size(0)

        # Flatten input
        x = coords_initial.view(B, -1)
        feat = self.mlp(x)

        # Predict components
        root_offset = self.root_offset_head(feat)  # (B, 2)
        du = self.bone_direction_head(feat).view(B, 13, 2)  # (B, 13, 2)
        ds = self.bone_length_head(feat)  # (B, 13)

        # 1. Compute physical bone lengths with tightly bounded scaling [0.8, 1.2]
        # L = avg_length * (1.0 + 0.2 * tanh(ds))
        scaling = 1.0 + 0.2 * torch.tanh(ds)  # (B, 13)
        L_bone = scaling * self.avg_lengths.unsqueeze(0)  # (B, 13)

        # 2. Compute refined unit direction vectors from initial and predicted offsets
        u_refined_list = []
        for i, (parent, child, _) in enumerate(self.BONES):
            # Initial direction from soft-argmax coordinates
            v_init = coords_initial[:, child, :] - coords_initial[:, parent, :]
            u_init = F.normalize(v_init, p=2, dim=-1, eps=1e-6)

            # Refined unit direction: Normalize(u_init + du)
            u_ref = F.normalize(u_init + du[:, i, :], p=2, dim=-1, eps=1e-6)
            u_refined_list.append(u_ref)

        u_refined = torch.stack(u_refined_list, dim=1)  # (B, 13, 2)

        # 3. Recursive Reconstruction starting from root (Neck/Thorax, joint 12)
        coords_refined = [None] * 14
        coords_refined[12] = coords_initial[:, 12, :] + root_offset

        for i, (parent, child, _) in enumerate(self.BONES):
            p_coord = coords_refined[parent]
            u = u_refined[:, i, :]
            L = L_bone[:, i].unsqueeze(-1)  # (B, 1)
            coords_refined[child] = p_coord + u * L

        coords_refined = torch.stack(coords_refined, dim=1)  # (B, 14, 2)

        return coords_refined


@register_model("kinematic_refined_hrnet")
class KinematicRefinedHRNet(nn.Module):
    """
    HRNet backbone with differentiable Kinematic Bone-Vector Decomposition refiner.
    Output heatmaps (standard) AND refined coordinates.
    """

    def __init__(self, config):
        super().__init__()

        # Extract sub-configs
        if "model" in config:
            model_cfg = config["model"]
            hrnet_cfg = model_cfg.get("hrnet", config)
        else:
            hrnet_cfg = config

        self.hrnet = HRNet(hrnet_cfg)
        self.soft_argmax = SoftArgmax2D(temperature=100.0)
        self.refiner = KinematicRefiner(
            hidden_dim=hrnet_cfg.get("kinematic_hidden_dim", 128)
        )

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x, return_refined=False):
        heatmaps = self.hrnet(x)

        # Initial coordinates from heatmaps via differentiable soft-argmax
        coords = self.soft_argmax(heatmaps)  # (B, 14, 2)

        # Refine via Kinematic Bone-Vector Decomposition
        refined_coords = self.refiner(coords)  # (B, 14, 2)

        if return_refined:
            return heatmaps, refined_coords

        return heatmaps
