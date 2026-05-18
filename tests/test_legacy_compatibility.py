import torch
import torch.nn as nn
import os
import pytest
from src.models.__init__ import load_model_for_inference


class LegacyHRNet(nn.Module):
    """Mock of the old HRNet structure to test remapping."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        # Old structure: stage2 -> modules_list -> 0 -> ...
        self.stage2 = nn.ModuleDict(
            {
                "modules_list": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "branches": nn.ModuleList(
                                    [nn.ModuleList([nn.Conv2d(32, 32, 3)])]
                                ),
                                "fusion": nn.ModuleDict(
                                    {
                                        "fuse_layers": nn.ModuleList(
                                            [nn.ModuleList([nn.Conv2d(32, 32, 1)])]
                                        )
                                    }
                                ),
                            }
                        )
                    ]
                )
            }
        )
        self.head = nn.Sequential(nn.Conv2d(480, 14, 1))


def test_legacy_remapping_logic():
    """Verify that the loader can map old 'modules_list' and 'fusion' paths to new structures."""
    # 1. Create a dummy state dict following the OLD structure
    old_state = {
        "conv1.weight": torch.randn(64, 1, 3, 3),
        "stage2.modules_list.0.branches.0.0.conv1.weight": torch.randn(32, 32, 3, 3),
        "stage2.modules_list.0.fusion.fuse_layers.0.1.0.weight": torch.randn(
            32, 64, 1, 1
        ),
        "head.0.weight": torch.randn(480, 480, 1, 1),
        "head.3.weight": torch.randn(14, 480, 1, 1),
        "head.3.bias": torch.randn(14),
    }

    checkpoint = {
        "model_state_dict": old_state,
        "config": {
            "model": {
                "name": "hrnet",
                "hrnet": {"num_joints": 14, "in_channels": 1, "architecture": "w32"},
            }
        },
    }

    ckpt_path = "scratch/legacy_mock.pth"
    os.makedirs("scratch", exist_ok=True)
    torch.save(checkpoint, ckpt_path)

    import src.models

    print(f"Registry in test: {list(src.models.MODEL_REGISTRY.keys())}")

    try:
        # 2. Try to load it into the CURRENT model
        # The current model uses stage2.0.branches... and stage2.0.fuse_layers.layers...
        model = load_model_for_inference(ckpt_path, device="cpu")

        # 3. Verify that the weights were actually copied
        current_state = model.state_dict()
        all_keys = list(current_state.keys())
        print(f"Total keys: {len(all_keys)}")
        print(f"Sample stage2 keys: {[k for k in all_keys if 'stage2' in k][:5]}")

        # Check mapping: stage2.modules_list.0.branches.0.0.conv1.weight -> stage2.0.branches.0.0.conv1.weight
        # Note: model. prefix comes from PoseDecodingWrapper
        assert torch.allclose(
            current_state["model.stage2.0.branches.0.0.conv1.weight"],
            old_state["stage2.modules_list.0.branches.0.0.conv1.weight"],
        )

        # Check mapping: fusion.fuse_layers -> fuse_layers.layers
        assert torch.allclose(
            current_state["model.stage2.0.fuse_layers.layers.0.1.0.weight"],
            old_state["stage2.modules_list.0.fusion.fuse_layers.0.1.0.weight"],
        )

        print("Legacy remapping test passed!")
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_loop2_loading_if_exists():
    """Verify loading of the real loop2_fixed_aug checkpoint if available."""
    loop2_path = "results/runs/loop2_fixed_aug/checkpoints/best_model.pth"
    if not os.path.exists(loop2_path):
        pytest.skip("loop2_fixed_aug checkpoint not found locally")

    # load_model_for_inference should not raise and should print 100% parity
    model = load_model_for_inference(loop2_path, device="cpu")
    assert model is not None
    assert hasattr(model, "output_type")
    assert model.output_type == "coordinates"


def test_coordinate_model_passthrough():
    """Verify that models already regressing coordinates (output_type='coordinates') are NOT double-decoded."""
    from src.models.registry import register_model

    @register_model("mock_coord_model")
    class MockCoordModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.param = nn.Parameter(torch.randn(1))

        @property
        def output_type(self):
            return "coordinates"

        def forward(self, x, **kwargs):
            # Returns joints (B, 14, 2) directly
            return torch.randn(x.shape[0], 14, 2)

    checkpoint = {
        "model_state_dict": {"param": torch.randn(1)},
        "config": {"model": {"name": "mock_coord_model", "mock_coord_model": {}}},
    }
    ckpt_path = "scratch/coord_mock.pth"
    torch.save(checkpoint, ckpt_path)

    try:
        from src.api.inference import InferenceService

        service = InferenceService()
        service.load_model(ckpt_path, force_reload=True)

        # This should NOT fail with shape mismatch
        img = torch.randn(1, 1, 256, 256)
        preds = service.predict(img)
        assert preds.shape == (1, 14, 2)
        print("Coordinate model passthrough test passed!")
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


if __name__ == "__main__":
    test_legacy_remapping_logic()
    test_coordinate_model_passthrough()
    test_loop2_loading_if_exists()
