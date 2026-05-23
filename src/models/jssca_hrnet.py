import torch
import torch.nn as nn
from .hrnet import HRNet
from .registry import register_model


class JointSpatialChannelAttention(nn.Module):
    """
    Backbone-Aware Joint-Spatial Neck Attention (JSSCA-v2 Option A).
    Downsamples the 480-channel backbone features to a compact 8x8 spatial grid,
    projects them to joint-specific spatial tokens, performs inter-joint and spatial
    co-attention across joints with learnable joint and spatial positional embeddings,
    and upsamples back to the original backbone feature shape.
    """

    def __init__(self, num_joints=14, in_channels=480, embed_dim=32, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.embed_dim = embed_dim  # Dimension per joint token (e.g. 32)
        self.joint_channels = num_joints * embed_dim  # e.g. 14 * 32 = 448

        # 1. Joint-Spatial Encoder
        # Maps (B, in_channels, 64, 64) -> (B, joint_channels, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Downsample 64x64 -> 32x32
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Downsample 32x32 -> 16x16
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Downsample 16x16 -> 8x8
            nn.Conv2d(
                128, self.joint_channels, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.joint_channels),
            nn.ReLU(inplace=True),
        )

        # Learnable positional embeddings to preserve anatomical identity and 2D coordinate positions
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, num_joints, 1, embed_dim) * 0.02
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, 8 * 8, embed_dim) * 0.02
        )

        # Multi-Head Attention layer to exchange geometric and spatial messages across joints
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Pre-LN LayerNorm and FFN layers for numerical stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 2. Progressive Deconvolutional Decoder
        # Maps (B, joint_channels, 8, 8) -> (B, in_channels, 64, 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.joint_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Upsample 8x8 -> 16x16
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Upsample 16x16 -> 32x32
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Upsample 32x32 -> 64x64
            nn.ConvTranspose2d(
                128, in_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, features):
        # features: (B, in_channels, 64, 64)
        B, C, H, W = features.shape

        # 1. Project and downsample features to joint representation
        # proj: (B, num_joints * embed_dim, 8, 8)
        proj = self.encoder(features)

        # 2. Reshape to joint-spatial tokens: (B, num_joints, 64, embed_dim)
        x = proj.reshape(B, self.num_joints, self.embed_dim, 8 * 8)
        x = x.permute(0, 1, 3, 2)  # (B, num_joints, 64, embed_dim)

        # 3. Add positional embeddings
        x = x + self.joint_pos_embed  # Broadcasts across spatial dimensions
        x = x + self.spatial_pos_embed  # Broadcasts across joints

        # 4. Flatten joint-spatial dimension for self-attention: (B, num_joints * 64, embed_dim)
        x_flat = x.reshape(B, self.num_joints * 64, self.embed_dim)
        
        # Pre-LN Self-Attention with residual connection
        x_norm1 = self.norm1(x_flat)
        attn_out, _ = self.mha(x_norm1, x_norm1, x_norm1)
        x_flat = x_flat + attn_out

        # Pre-LN FFN with residual connection
        x_norm2 = self.norm2(x_flat)
        ffn_out = self.ffn(x_norm2)
        refined_flat = x_flat + ffn_out

        # 5. Reshape back and decode
        refined = refined_flat.reshape(B, self.num_joints, 8, 8, self.embed_dim)
        refined = refined.permute(0, 1, 4, 2, 3)  # (B, num_joints, embed_dim, 8, 8)
        refined = refined.reshape(B, self.joint_channels, 8, 8)

        # Upsample back to feature space
        delta = self.decoder(refined)

        # Residual link
        return features + delta


@register_model("jssca_hrnet")
class JSSCAHRNet(nn.Module):
    """
    HRNet-W32 backbone refined with a Joint-Symmetric Spatial-Channel Attention (JSSCA-v2) stage.
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
        embed_dim = hrnet_cfg.get("jssca_embed_dim", 32)
        num_heads = hrnet_cfg.get("jssca_num_heads", 4)

        # In W32 HRNet, parallel streams sum to 480 channels
        in_channels = sum(self.hrnet.W32)

        self.jssca = JointSpatialChannelAttention(
            num_joints=num_joints,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x):
        # 1. Base features from HRNet backbone (returning features = True)
        _, features = self.hrnet(x, return_features=True)

        # 2. Refined features from JSSCA-v2 Neck Attention
        refined_features = self.jssca(features)

        # 3. Predict refined heatmaps from refined features
        refined_heatmaps = self.hrnet.head(refined_features)

        return refined_heatmaps
