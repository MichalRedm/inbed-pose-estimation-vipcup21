import torch
import torch.nn.functional as F
from src.models import build_model
from src.models.jssca_hrnet import JSSCAHRNet, JointSpatialChannelAttention, soft_argmax_2d, shift_heatmap


def test_soft_argmax_2d():
    """Verify soft-argmax correctly locates a single Gaussian peak and is differentiable."""
    heatmaps = torch.zeros(1, 1, 64, 64, requires_grad=True)
    # Put a peak at center (32, 32)
    heatmaps.data[0, 0, 32, 32] = 10.0
    
    coords = soft_argmax_2d(heatmaps, temperature=10.0)
    assert coords.shape == (1, 1, 2), f"Expected (1, 1, 2), got {coords.shape}"
    
    # Coordinate at center should be close to 0 in [-1, 1] range
    assert abs(coords[0, 0, 0].item()) < 0.05
    assert abs(coords[0, 0, 1].item()) < 0.05
    
    # Check gradients flow back to heatmaps
    loss = coords.sum()
    loss.backward()
    assert heatmaps.grad is not None, "Gradients must flow back to heatmaps"


def test_shift_heatmap():
    """Verify shift_heatmap correctly translates a peak in the expected direction."""
    # Create a 64x64 heatmap with a single sharp peak at center (32, 32)
    heatmaps = torch.zeros(1, 1, 64, 64)
    heatmaps[0, 0, 32, 32] = 1.0
    
    # Shift right by 0.1 and down by 0.2 (in normalized coordinates [-1, 1])
    # 0.1 shift in [-1, 1] range is 0.1 * 32 = 3.2 pixels right
    # 0.2 shift is 0.2 * 32 = 6.4 pixels down
    offsets = torch.tensor([[0.1, 0.2]]) # [dx, dy]
    
    shifted = shift_heatmap(heatmaps, offsets)
    
    # Verify shape
    assert shifted.shape == (1, 1, 64, 64)
    
    # Find new peak coordinate using standard argmax
    flat = shifted.view(-1)
    new_idx = flat.argmax().item()
    new_y = new_idx // 64
    new_x = new_idx % 64
    
    # Original center was (32, 32). New center should be shifted right (x increase) and down (y increase)
    assert new_x > 32, f"Expected shift right (x > 32), got x = {new_x}"
    assert new_y > 32, f"Expected shift down (y > 32), got y = {new_y}"


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
    test_soft_argmax_2d()
    test_shift_heatmap()
    test_jssca_block_shape()
    test_jssca_hrnet_shape()
    print("All JSSCA-v5 unit tests passed successfully!")
