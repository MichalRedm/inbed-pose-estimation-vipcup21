import torch
import numpy as np
from src.utils.pose import decode_heatmaps

def test_decoding_parity():
    """Verify that argmax and soft-argmax produce expected results for a sharp Gaussian."""
    B, J, H, W = 1, 1, 64, 64
    image_size = (256, 256)
    
    # Create a sharp peak at (32, 16) in 64x64 heatmap
    # Image space: (32 * 4, 16 * 4) = (128, 64)
    heatmaps = torch.zeros((B, J, H, W))
    heatmaps[0, 0, 16, 32] = 10.0 # High value
    
    # Argmax decoding
    preds_argmax = decode_heatmaps(heatmaps, image_size, method="argmax")
    assert preds_argmax[0, 0, 0] == 128.0
    assert preds_argmax[0, 0, 1] == 64.0
    
    # Soft-argmax decoding
    # With a very high peak, soft-argmax should be extremely close to the peak
    preds_soft = decode_heatmaps(heatmaps, image_size, method="soft-argmax")
    np.testing.assert_allclose(preds_soft[0, 0, 0].item(), 128.0, atol=1.0)
    np.testing.assert_allclose(preds_soft[0, 0, 1].item(), 64.0, atol=1.0)

def test_soft_argmax_subpixel():
    """Verify that soft-argmax can recover sub-pixel coordinates."""
    B, J, H, W = 1, 1, 64, 64
    image_size = (64, 64) # Use 1:1 scale for simplicity
    
    # Create two peaks of equal value at (32, 32) and (33, 32)
    # Expected soft-argmax: (32.5, 32)
    heatmaps = torch.zeros((B, J, H, W))
    heatmaps[0, 0, 32, 32] = 10.0
    heatmaps[0, 0, 32, 33] = 10.0
    
    preds = decode_heatmaps(heatmaps, image_size, method="soft-argmax")
    
    # (32.5, 32)
    assert 32.4 < preds[0, 0, 0] < 32.6
    assert 31.9 < preds[0, 0, 1] < 32.1

if __name__ == "__main__":
    test_decoding_parity()
    test_soft_argmax_subpixel()
    print("API parity tests passed!")
