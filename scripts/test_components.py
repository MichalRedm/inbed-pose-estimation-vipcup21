import torch
from src.data.dataset import VIPCupDataset
from src.models.hrnet import get_pose_net
from src.utils import load_config


def test_components():
    config = load_config()
    model_cfg = config.get("model", {}).get("hrnet", {})

    print("Testing Model Instantiation...")
    model = get_pose_net(model_cfg)
    print(
        f"Model created. Total parameters: {sum(p.numel() for p in model.parameters())}"
    )

    print("\nTesting Model Forward Pass (Dummy Batch)...")
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [1, 14, 64, 64])")

    print("\nTesting Dataset Instantiation (Empty Path Test)...")
    try:
        # This will likely find 0 samples since data/raw is empty, but we check if it crashes
        dataset = VIPCupDataset(root="data/raw", subjects=[1])
        print(f"Dataset instantiated. Found {len(dataset)} samples.")
    except Exception as e:
        print(f"Dataset instantiation failed (as expected if folders missing): {e}")


if __name__ == "__main__":
    test_components()
