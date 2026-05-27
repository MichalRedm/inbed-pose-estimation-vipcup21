import torch
from src.models.cyclegan.generator import GeneratorResNet
from src.models.cyclegan.cut_loss import PatchSampleF, PatchNCELoss

def test_cut_forward_backward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    input_shape = (3, 256, 256)
    
    # Initialize Models
    G = GeneratorResNet(input_shape, num_residual_blocks=9, pretrained=False).to(device)
    F = PatchSampleF().to(device)
    nce_loss = PatchNCELoss().to(device)
    
    # Dummy tensors representing batch size of 2
    real_A = torch.randn(2, 3, 256, 256, device=device)
    
    # 1. Forward G to get generated image and encoder features
    fake_B, feat_k = G(real_A, return_features=True)
    assert len(feat_k) == 5, f"Expected 5 feature maps, got {len(feat_k)}"
    
    # Detach features from real image (target for InfoNCE)
    feat_k = [f.detach() for f in feat_k]
    
    # 2. Extract features from fake_B
    _, feat_q = G(fake_B, return_features=True)
    assert len(feat_q) == 5, f"Expected 5 feature maps, got {len(feat_q)}"
    
    # 3. Apply PatchSampleF
    num_patches = 64  # Small subset for testing
    pool_q, patch_ids = F(feat_q, num_patches=num_patches)
    pool_k, _ = F(feat_k, patch_ids=patch_ids)
    
    assert len(pool_q) == 5
    assert pool_q[0].shape == (2, num_patches, 256), f"Shape is {pool_q[0].shape}"
    assert pool_k[0].shape == (2, num_patches, 256), f"Shape is {pool_k[0].shape}"
    
    # 4. Compute NCE loss
    loss = nce_loss(pool_q, pool_k)
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Calculated InfoNCE Loss: {loss.item()}")
    
    # 5. Backward Pass
    loss.backward()
    
    # Check Gradients
    # MLP projector should get gradients
    assert F.mlps[0][0].weight.grad is not None, "No gradients in F"
    
    # Encoder should get gradients through feat_q (i.e. from the generation pass)
    assert G.encoder[1].weight.grad is not None, "No gradients in G encoder"
    
    print("SUCCESS: CUT Forward and Backward pass completed without NaNs. Gradients propagated correctly.")

if __name__ == "__main__":
    test_cut_forward_backward()
