import torch
from src.models import build_model
from src.models.jssca_hrnet import JSSCAHRNet, JointSpatialChannelAttention


def test_jssca_block_shape():
    """Verify JointSpatialChannelAttention post-processing block processes heatmaps correctly."""
    block = JointSpatialChannelAttention(
        num_joints=14,
        embed_dim=256,
        num_heads=4,
    )
    block.eval()
    x = torch.zeros(2, 14, 64, 64)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 14, 64, 64), f"Expected (2, 14, 64, 64), got {out.shape}"


def test_jssca_hrnet_shape():
    """Verify JSSCAHRNet processes input image tensor correctly and outputs 14 heatmaps."""
    config = {
        "model": {
            "name": "jssca_hrnet",
            "jssca_hrnet": {
                "pretrained": False,
                "architecture": "w32",
                "num_joints": 14,
                "in_channels": 3,
                "heatmap_size": [64, 64],
                "jssca_embed_dim": 256,
                "jssca_num_heads": 4,
            },
        }
    }
    model = build_model(config)
    assert isinstance(model, JSSCAHRNet)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 14, 64, 64), f"Expected (2, 14, 64, 64), got {out.shape}"


if __name__ == "__main__":
    test_jssca_block_shape()
    test_jssca_hrnet_shape()
    print("All JSSCA-v4 unit tests passed successfully!")

