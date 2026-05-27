import torch
import unittest
from src.models import build_model
from src.utils import load_config


class TestProjectInfrastructure(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_model_creation(self):
        """Test if the model can be instantiated with default config via factory."""
        model = build_model(self.config)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        """Test if the model forward pass produces correct heatmap dimensions."""
        model = build_model(self.config)
        model_name = self.config.get("model", {}).get("name")
        in_channels = (
            self.config.get("model", {}).get(model_name, {}).get("in_channels", 1)
        )
        dummy_input = torch.randn(1, in_channels, 256, 256)
        output = model(dummy_input)
        # Expected Output: [Batch, Joints, HeatmapH, HeatmapW]
        # In our implementation, input 256 -> output 64
        self.assertEqual(output.shape, (1, 14, 64, 64))


if __name__ == "__main__":
    unittest.main()
