import torch
import torch.nn as nn
from .hrnet import HRNet
from .registry import register_model


class JointSpatialChannelAttention(nn.Module):
    """
    Joint-Symmetric Spatial-Channel Attention (JSSCA-v3 Post-Processor).
    Downsamples the predicted heatmaps (14 channels) to a compact 8x8 spatial grid,
    projects them to joint-specific spatial tokens, performs inter-joint and spatial
    co-attention across joints with learnable joint and spatial positional embeddings,
    and upsamples back to 64x64 using a progressive deconvolutional decoder
    with multi-scale residual skip connections to preserve coordinate anchors.
    """

    def __init__(self, num_joints=14, embed_dim=32, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim  # Dimension per joint token (e.g. 32)
        self.joint_channels = num_joints * embed_dim  # e.g. 14 * 32 = 448

        # 1. Joint-wise U-Net Encoder
        # We process each joint's (1, 64, 64) heatmap independently down to (embed_dim, 8, 8)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc_down3 = nn.Sequential(
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
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

        # 2. Progressive Deconvolutional Decoder with skip connections
        # Maps (B*J, embed_dim, 8, 8) -> (B*J, 1, 64, 64)
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 32, kernel_size=4, stride=2, padding=1, bias=False),
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

        h1 = self.enc_conv(h0)      # (B*J, 16, 64, 64)
        h2 = self.enc_down1(h1)     # (B*J, 16, 32, 32)
        h3 = self.enc_down2(h2)     # (B*J, 32, 16, 16)
        proj = self.enc_down3(h3)   # (B*J, embed_dim, 8, 8)

        # 2. Reshape to joint-spatial tokens: (B, num_joints, 64, embed_dim)
        x = proj.reshape(B, self.num_joints, self.embed_dim, 8 * 8)
        x = x.permute(0, 1, 3, 2)  # (B, num_joints, 64, embed_dim)

        # 3. Add positional embeddings
        x = x + self.joint_pos_embed  # Broadcasts across spatial dimensions (64)
        x = x + self.spatial_pos_embed  # Broadcasts across joints (14)

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

        # 5. Reshape back for decoding: (B*J, embed_dim, 8, 8)
        refined = refined_flat.reshape(B, self.num_joints, 8, 8, self.embed_dim)
        refined = refined.permute(0, 1, 4, 2, 3)  # (B, num_joints, embed_dim, 8, 8)
        refined = refined.reshape(B * self.num_joints, self.embed_dim, 8, 8)

        # 6. Progressive upsampling with multi-scale skip connections
        d1 = self.dec_up1(refined)  # (B*J, 32, 16, 16)
        d1 = d1 + h3                # Skip connection at 16x16

        d2 = self.dec_up2(d1)       # (B*J, 16, 32, 32)
        d2 = d2 + h2                # Skip connection at 32x32

        delta = self.dec_up3(d2)    # (B*J, 1, 64, 64)

        # 7. Global residual skip connection
        out = h0 + delta            # (B*J, 1, 64, 64)
        
        return out.reshape(B, J, H, W)


@register_model("jssca_hrnet")
class JSSCAHRNet(nn.Module):
    """
    HRNet-W32 backbone refined with a Joint-Symmetric Spatial-Channel Attention (JSSCA-v3) stage.
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

        # 2. Refined heatmaps from JSSCA-v3 Post-Processor
        refined_heatmaps = self.jssca(heatmaps)

        return refined_heatmaps
