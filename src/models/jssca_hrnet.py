import torch
import torch.nn as nn
import torch.nn.functional as F
from .hrnet import HRNet
from .registry import register_model


def soft_argmax_2d(heatmaps, temperature=100.0):
    """
    Differentiable 2D soft-argmax.
    heatmaps: (B, J, H, W)
    Returns: coordinates of shape (B, J, 2) in normalized [-1, 1] range.
    """
    B, J, H, W = heatmaps.shape
    device = heatmaps.device
    flat = heatmaps.view(B, J, -1)
    probs = torch.softmax(flat * temperature, dim=-1).view(B, J, H, W)

    # meshgrid in range [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    x = torch.sum(probs * grid_x.unsqueeze(0).unsqueeze(0), dim=(2, 3))
    y = torch.sum(probs * grid_y.unsqueeze(0).unsqueeze(0), dim=(2, 3))
    return torch.stack([x, y], dim=-1) # (B, J, 2)


def shift_heatmap(heatmaps, offsets):
    """
    Differentiably shifts heatmaps using 2D grid sampling.
    heatmaps: (B * J, 1, H, W)
    offsets: (B * J, 2) in normalized coordinate range [-1, 1].
    """
    BJ, C, H, W = heatmaps.shape
    device = heatmaps.device

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(BJ, 1, 1, 1) # (BJ, H, W, 2)
    grid = grid - offsets.view(BJ, 1, 1, 2)

    return F.grid_sample(
        heatmaps, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


class JointSpatialChannelAttention(nn.Module):
    """
    Joint-Symmetric Spatial-Channel Attention (JSSCA-v5 Spatially-Anchored Post-Processor).
    Uses a highly focused 14-joint attention bottleneck anchored with precise 2D joint coordinates.
    Differentiably shifts the high-resolution heatmaps using predicted coordinate offsets and adds
    local residual heatmaps, completely bypassing low-resolution upsampling and transposed conv bottlenecks.
    """

    def __init__(self, num_joints=14, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # 1. Semantic Joint Encoder
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

        # 2. Coordinate Joint Encoder (accepts 3D: [x, y, confidence])
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim),
        )

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

        # 3. Refinement Predictors
        # Predicts 2D coordinate shift in range [-0.5, 0.5] (regularized by tanh)
        self.coord_predictor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        # Predicts local residual heatmap changes (64x64)
        self.residual_decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 64),
        )

    def forward(self, heatmaps):
        # heatmaps: (B, num_joints, 64, 64)
        B, J, H, W = heatmaps.shape
        device = heatmaps.device

        # 1. Extract 2D Coordinate anchors via differentiable Soft-Argmax
        # P: (B, J, 2)
        P = soft_argmax_2d(heatmaps, temperature=100.0)
        # Extract peak confidence for each joint (B, J) to act as a gating factor
        conf = heatmaps.view(B, J, -1).max(dim=-1)[0]
        # Concatenate 2D coordinates and peak confidence to form 3D anchors: (B, J, 3)
        P_anchor = torch.cat([P, conf.unsqueeze(-1)], dim=-1)

        # 2. Reshape to process each joint's heatmap independently through the semantic encoder
        h0 = heatmaps.reshape(B * J, 1, H, W)
        h4 = self.enc_conv(h0)      # (B*J, 64, 8, 8)

        # Global average pool to get joint-specific activation signature
        pooled = self.pool(h4).squeeze(-1).squeeze(-1)  # (B*J, 64)
        semantic_embeddings = self.proj(pooled)         # (B*J, embed_dim)
        semantic_embeddings = semantic_embeddings.reshape(B, J, self.embed_dim) # (B, J, embed_dim)

        # 3. Project 3D coordinate-confidence anchors to embedding space
        coord_embeddings = self.coord_encoder(P_anchor)  # (B, J, embed_dim)
        # Softly gate coordinate embeddings by peak confidence
        coord_embeddings = coord_embeddings * conf.unsqueeze(-1)

        # 4. Combine semantic, coordinate, and joint positional embeddings
        x = semantic_embeddings + coord_embeddings + self.joint_pos_embed

        # 5. Perform Spatially-Anchored Joint Self-Attention
        x_norm1 = self.norm1(x)
        attn_out, _ = self.mha(x_norm1, x_norm1, x_norm1)
        x = x + attn_out

        # Pre-LN FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        refined = x + ffn_out  # (B, J, embed_dim)

        # 6. Predict coordinate shifts and residual heatmaps
        refined_flat = refined.reshape(B * J, self.embed_dim)

        # Regularized coordinate shift offsets: bounded to [-0.5, 0.5] range (tanh)
        offsets = torch.tanh(self.coord_predictor(refined_flat)) * 0.5  # (B*J, 2)

        # Local residual heatmap changes
        residuals = self.residual_decoder(refined_flat).reshape(B * J, 1, H, W)  # (B*J, 1, H, W)

        # 7. Apply differentiable grid translation
        h0_shifted = shift_heatmap(h0, offsets)  # (B*J, 1, H, W)

        # 8. Add local residual correction
        out = h0_shifted + residuals

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

