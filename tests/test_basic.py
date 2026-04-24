import torch
import unittest
from src.models.hrnet import get_pose_net
from src.utils import load_config


class TestProjectInfrastructure(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.model_cfg = self.config.get("model", {}).get("hrnet", {})

    def test_model_creation(self):
        """Test if the HRNet model can be instantiated with default config."""
        model = get_pose_net(self.model_cfg)
        self.assertIsNotNone(model)

    def test_model_forward(self):
        """Test if the model forward pass produces correct heatmap dimensions."""
        model = get_pose_net(self.model_cfg)
        dummy_input = torch.randn(1, 1, 256, 256)
        output = model(dummy_input)
        # Expected Output: [Batch, Joints, HeatmapH, HeatmapW]
        # In our simplified HRNet, input 256 -> two 2-stride convs -> output 64
        self.assertEqual(output.shape, (1, 14, 64, 64))


if __name__ == "__main__":
    unittest.main()
