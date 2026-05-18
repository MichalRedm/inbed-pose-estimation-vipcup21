import torch
import numpy as np
import random
from PIL import Image
from src.data.augmentations import (
    DataAugmenter,
    ThermalIntensityJitter,
    IRSensorNoise,
    CutoutAugmentation
)

def test_thermal_intensity_jitter():
    """
    Test that ThermalIntensityJitter successfully alters the image,
    respects the ranges, and doesn't crash on either PIL or Tensor inputs.
    """
    # 1. PIL Test
    img_pil = Image.new("L", (128, 128), 128)
    jitter = ThermalIntensityJitter(probability=1.0, intensity_range=(0.55, 1.15), contrast_range=(0.5, 1.15))
    out_pil = jitter(img_pil)
    assert isinstance(out_pil, Image.Image)
    assert out_pil.size == (128, 128)
    
    # Verify the pixel values were modified
    arr_in = np.array(img_pil)
    arr_out = np.array(out_pil)
    assert not np.array_equal(arr_in, arr_out), "Photometric jitter did not modify PIL pixels"

    # 2. Tensor Test
    img_tensor = torch.full((1, 128, 128), 0.5)
    out_tensor = jitter(img_tensor)
    assert torch.is_tensor(out_tensor)
    assert out_tensor.shape == (1, 128, 128)
    assert not torch.equal(img_tensor, out_tensor), "Photometric jitter did not modify Tensor pixels"
    assert torch.all(out_tensor >= 0.0) and torch.all(out_tensor <= 1.0)


def test_ir_sensor_noise():
    """
    Test that IRSensorNoise injects noise without crash,
    for both PIL and Tensor formats.
    """
    img_pil = Image.new("L", (128, 128), 128)
    noise_injector = IRSensorNoise(probability=1.0, noise_sigma=(5, 12), dead_pixel_ratio=0.01)
    out_pil = noise_injector(img_pil)
    assert isinstance(out_pil, Image.Image)
    assert out_pil.size == (128, 128)
    
    arr_in = np.array(img_pil)
    arr_out = np.array(out_pil)
    assert not np.array_equal(arr_in, arr_out), "Noise injection did not modify PIL pixels"

    # Tensor Test
    img_tensor = torch.full((1, 128, 128), 0.5)
    out_tensor = noise_injector(img_tensor)
    assert torch.is_tensor(out_tensor)
    assert out_tensor.shape == (1, 128, 128)
    assert not torch.equal(img_tensor, out_tensor), "Noise injection did not modify Tensor pixels"
    assert torch.all(out_tensor >= 0.0) and torch.all(out_tensor <= 1.0)


def test_cutout_augmentation():
    """
    Test that CutoutAugmentation correctly blacks out a region of the image
    for both PIL and Tensor formats.
    """
    # 1. PIL Test
    img_pil = Image.new("L", (128, 128), 128)
    cutout = CutoutAugmentation(probability=1.0, size_ratio=0.35)
    out_pil = cutout(img_pil)
    assert isinstance(out_pil, Image.Image)
    
    arr_out = np.array(out_pil)
    # Cutout puts zeros (black)
    assert np.any(arr_out == 0), "Cutout did not produce any zero/black pixels"
    assert np.any(arr_out == 128), "Cutout over-erased entire image"

    # 2. Tensor Test
    img_tensor = torch.full((1, 128, 128), 0.5)
    out_tensor = cutout(img_tensor)
    assert torch.is_tensor(out_tensor)
    assert torch.any(out_tensor == 0.0), "Cutout did not produce any zero pixels in Tensor"
    assert torch.any(out_tensor == 0.5), "Cutout over-erased entire Tensor image"


def test_data_augmenter_integration():
    """
    Verify integration inside DataAugmenter when calling __call__.
    """
    config = {
        "enabled": True,
        "flip_prob": 0.5,
        "rotation_range": [-30, 30],
        "scaling_range": [0.8, 1.2],
        "translation": [0.10, 0.10],
        "cutout_prob": 1.0,
        "cutout_size_ratio": 0.35,
        "intensity_jitter_prob": 1.0,
        "intensity_jitter_range": [0.55, 1.15],
        "contrast_jitter_range": [0.5, 1.15],
        "sensor_noise_prob": 1.0,
        "sensor_noise_sigma": [5, 12]
    }
    
    img = Image.new("L", (256, 256), 128)
    joints = np.zeros((3, 14))
    joints[0, :] = 100.0
    joints[1, :] = 120.0
    joints[2, :] = 0  # visible

    augmenter = DataAugmenter(config=config, is_training=True)
    out_img, out_joints = augmenter(img, joints, is_ir=True)
    
    assert isinstance(out_img, Image.Image) or torch.is_tensor(out_img)
    assert out_joints is not None
    assert out_joints.shape == (3, 14)
    
    # Check that pixels are modified
    if isinstance(out_img, Image.Image):
        arr_out = np.array(out_img)
        assert np.any(arr_out == 0), "Integrated cutout failed"
    else:
        assert torch.any(out_img == 0.0), "Integrated cutout failed"


if __name__ == "__main__":
    print("Running extended augmentation tests...")
    test_thermal_intensity_jitter()
    test_ir_sensor_noise()
    test_cutout_augmentation()
    test_data_augmenter_integration()
    print("All extended augmentation tests passed successfully!")
