import torch
import torch.nn as nn
from .hrnet import HRNet
from .registry import register_model


class JointSpatialChannelAttention(nn.Module):
    """
    Joint-Symmetric Spatial-Channel Attention (JSSCA-v4 No Skips Post-Processor).
    Uses a highly focused 14-joint attention bottleneck (sequence length = 14)
    to eliminate visual token dilution. Reconstructs refined coordinate heatmaps
    using a progressive deconvolutional decoder without intermediate skips to prevent
    degenerate gradient shortcuts, forcing 100% of the information to flow through
    the self-attention layers.
    """

    def __init__(self, num_joints=14, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # 1. Joint-wise Encoder
        # Downsamples each joint's (1, 64, 64) heatmap to (64, 8, 8)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 64x64 -> 32x32
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, embed_dim)

        # Learnable Joint Positional Embeddings to preserve anatomical identities
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
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
        # Maps (B*J, embed_dim) -> (B*J, 1, 64, 64)
        self.proj_back = nn.Linear(embed_dim, 64 * 8 * 8)
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.dec_up3 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, heatmaps):
        # heatmaps: (B, num_joints, 64, 64)
        B, J, H, W = heatmaps.shape

        # 1. Reshape to process each joint's heatmap independently through the encoder
        h0 = heatmaps.reshape(B * J, 1, H, W)

        h4 = self.enc_conv(h0)      # (B*J, 64, 8, 8)

        # Global average pool to get joint-specific activation signature
        pooled = self.pool(h4).squeeze(-1).squeeze(-1)  # (B*J, 64)
        proj = self.proj(pooled)                        # (B*J, embed_dim)

        # 2. Reshape back to sequence of joints: (B, num_joints, embed_dim)
        x = proj.reshape(B, self.num_joints, self.embed_dim)

        # 3. Add joint positional embeddings (sequence length = 14)
        x = x + self.joint_pos_embed

        # 4. Perform highly focused Self-Attention across joints
        x_norm1 = self.norm1(x)
        attn_out, _ = self.mha(x_norm1, x_norm1, x_norm1)
        x = x + attn_out

        # Pre-LN FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        refined = x + ffn_out  # (B, num_joints, embed_dim)

        # 5. Reshape back to joint-wise representations for decoding
        refined = refined.reshape(B * self.num_joints, self.embed_dim)
        d0 = self.proj_back(refined)                    # (B*J, 64 * 8 * 8)
        d0 = d0.reshape(B * self.num_joints, 64, 8, 8)

        # 6. Progressive upsampling
        d1 = self.dec_up1(d0)       # (B*J, 32, 16, 16)
        d2 = self.dec_up2(d1)       # (B*J, 16, 32, 32)
        delta = self.dec_up3(d2)    # (B*J, 1, 64, 64)

        # 7. Global residual skip connection
        out = h0 + delta            # (B*J, 1, 64, 64)

        return out.reshape(B, J, H, W)


@register_model("jssca_hrnet")
class JSSCAHRNet(nn.Module):
    """
    HRNet-W32 backbone refined with a Joint-Symmetric Spatial-Channel Attention (JSSCA-v4) stage.
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
        embed_dim = hrnet_cfg.get("jssca_embed_dim", 256)
        num_heads = hrnet_cfg.get("jssca_num_heads", 4)

        self.jssca = JointSpatialChannelAttention(
            num_joints=num_joints,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x):
        # 1. Base heatmaps from HRNet backbone
        heatmaps = self.hrnet(x)

        # 2. Refined heatmaps from JSSCA-v4 Post-Processor
        refined_heatmaps = self.jssca(heatmaps)

        return refined_heatmaps

