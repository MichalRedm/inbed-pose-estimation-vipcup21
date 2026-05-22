import torch
import torch.nn as nn
from .hrnet import HRNet
from .registry import register_model


class JointSpatialChannelAttention(nn.Module):
    """
    Joint-Symmetric Spatial-Channel Attention (JSSCA) block.
    Compresses individual 2D joint heatmaps into compact 1D latent embeddings,
    performs multi-head self-attention across joints with learnable joint positional
    embeddings to exchange anatomical messages, and decodes back to refined 2D heatmaps.
    """

    def __init__(
        self, num_joints=14, heatmap_size=(64, 64), embed_dim=256, num_heads=4
    ):
        super().__init__()
        self.num_joints = num_joints
        self.h, self.w = heatmap_size
        self.flat_dim = self.h * self.w
        self.embed_dim = embed_dim

        # Encoder to project each joint's 2D heatmap into a 1D latent embedding
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (B*J, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (B*J, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B*J, 32, 1, 1)
            nn.Flatten(),  # -> (B*J, 32)
            nn.Linear(32, embed_dim),  # -> (B*J, embed_dim)
        )

        # Learnable Joint Positional Embeddings to preserve anatomical identities
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )

        # Multi-Head Attention layer to exchange geometric and spatial messages across joints
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Decoder to project refined joint embeddings back to 2D heatmaps
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.flat_dim),
        )

    def forward(self, heatmaps):
        # heatmaps: (B, J, H, W)
        B, J, H, W = heatmaps.shape

        # 1. Reshape heatmaps to run through the joint-wise encoder
        x = heatmaps.view(B * J, 1, H, W)  # (B*J, 1, H, W)
        embeddings = self.encoder(x)  # (B*J, embed_dim)

        # 2. Reshape back to batch and add joint positional embeddings
        embeddings = embeddings.view(B, J, self.embed_dim)  # (B, J, embed_dim)
        embeddings_with_pos = embeddings + self.joint_pos_embed

        # 3. Apply Multi-Head Self-Attention across joints
        # query, key, value all come from embeddings_with_pos
        refined_embeddings, _ = self.mha(
            embeddings_with_pos, embeddings_with_pos, embeddings_with_pos
        )  # (B, J, embed_dim)

        # 4. Reshape and decode back to heatmaps
        refined_embeddings = refined_embeddings.view(
            B * J, self.embed_dim
        )  # (B*J, embed_dim)
        delta = self.decoder(refined_embeddings)  # (B*J, H*W)
        delta = delta.view(B, J, H, W)  # (B, J, H, W)

        # 5. Residual link
        return heatmaps + delta


@register_model("jssca_hrnet")
class JSSCAHRNet(nn.Module):
    """
    HRNet-W32 backbone refined with a Joint-Symmetric Spatial-Channel Attention (JSSCA) stage.
    Outputs: refined high-resolution heatmaps.
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

        num_joints = hrnet_cfg.get("num_joints", 14)
        heatmap_size = tuple(hrnet_cfg.get("heatmap_size", [64, 64]))
        embed_dim = hrnet_cfg.get("jssca_embed_dim", 256)
        num_heads = hrnet_cfg.get("jssca_num_heads", 4)

        self.jssca = JointSpatialChannelAttention(
            num_joints=num_joints,
            heatmap_size=heatmap_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x):
        # 1. Base heatmaps from HRNet
        heatmaps = self.hrnet(x)

        # 2. Refined heatmaps from JSSCA
        refined_heatmaps = self.jssca(heatmaps)

        return refined_heatmaps
