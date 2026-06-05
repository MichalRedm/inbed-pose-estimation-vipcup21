import torch
from src.models.cyclegan.generator import GeneratorResNet
from src.models.cyclegan.cut_loss import PatchSampleF, PatchNCELoss


def test_cut_forward_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    print(
        "SUCCESS: CUT Forward and Backward pass completed without NaNs. Gradients propagated correctly."
    )


def test_cut_lightning_module():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_shape = (3, 256, 256)
    G = GeneratorResNet(input_shape, num_residual_blocks=9, pretrained=False).to(device)
    from src.models.cyclegan.discriminator import Discriminator
    D = Discriminator(input_shape).to(device)
    F_net = PatchSampleF().to(device)
    
    import torch.nn as nn
    P = nn.Conv2d(3, 14, 1).to(device) # dummy pose estimator
    P.eval()
    
    opt_G = torch.optim.Adam(list(G.parameters()) + list(F_net.parameters()), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    
    from src.training.cut_trainer import CUTLightningModule
    module = CUTLightningModule(
        G=G, D=D, F=F_net,
        optimizer_G=opt_G, optimizer_D=opt_D,
        config={"training": {"lambda_pose": 1.0}},
        pose_estimator=P
    )
    module.to(device)
    
    real_A = torch.randn(2, 3, 256, 256, device=device)
    real_B = torch.randn(2, 3, 256, 256, device=device)
    batch = (real_A, real_B)
    
    # fake trainer setup
    class FakeTrainer:
        def __init__(self):
            self.strategy = True
            self.barebones = False
            self._results = None
    module._trainer = FakeTrainer()
    module.optimizers = lambda: (opt_G, opt_D)
    module.log = lambda *args, **kwargs: None
    
    try:
        module.training_step(batch, 0)
        print("SUCCESS: CUTLightningModule training_step completed without error.")
    except Exception as e:
        print(f"FAILED: CUTLightningModule error: {e}")
        raise e

if __name__ == "__main__":
    test_cut_forward_backward()
    test_cut_lightning_module()
