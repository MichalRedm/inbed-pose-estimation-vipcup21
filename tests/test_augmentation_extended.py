import numpy as np
from PIL import Image
from src.data.augmentations import (
    CutoutAugmentation,
    ThermalIntensityJitter,
    IRSensorNoise,
    DataAugmenter,
)


def test_cutout_augmentation():
    # Create a 256x256 gray PIL Image
    img = Image.fromarray(np.ones((256, 256), dtype=np.uint8) * 128)

    # Run cutout with 100% probability
    cutout = CutoutAugmentation(probability=1.0, size_ratio=0.35)
    img_aug = cutout(img)

    img_np = np.array(img_aug)

    # Black rectangle should have value 0
    zeros = np.sum(img_np == 0)
    assert zeros > 0, "Cutout should zero out some pixels"

    # Cutout should not exceed size_ratio
    max_pixels = int(256 * 0.35) * int(256 * 0.35)
    assert zeros <= max_pixels, "Cutout area is too large"

    # Try with 0% probability
    cutout_off = CutoutAugmentation(probability=0.0)
    img_aug_off = cutout_off(img)
    assert np.array_equal(np.array(img_aug_off), np.array(img)), (
        "Probability 0 should make no change"
    )


def test_intensity_jitter():
    img = Image.fromarray(np.ones((256, 256), dtype=np.uint8) * 100)

    # Apply jitter with 100% prob
    jitter = ThermalIntensityJitter(
        probability=1.0,
        brightness_range=[0.6, 0.9],  # strictly attenuate brightness
        contrast_range=[0.5, 0.8],
    )
    img_aug = jitter(img)
    img_np = np.array(img_aug)

    # Attenuated brightness should make pixels strictly darker
    assert np.mean(img_np) < 100.0, "Intensity jitter brightness scale failed"
    assert np.all(img_np >= 0) and np.all(img_np <= 255), (
        "Values must remain in bounds [0, 255]"
    )


def test_sensor_noise():
    img = Image.fromarray(np.ones((256, 256), dtype=np.uint8) * 128)

    # Apply noise with 100% prob
    noise = IRSensorNoise(probability=1.0, sigma_range=[10.0, 15.0], sp_prob=0.01)
    img_aug = noise(img)
    img_np = np.array(img_aug)

    # The original image was flat (std = 0)
    # The augmented image must have high variance due to Gaussian noise
    assert np.std(img_np) > 5.0, "Sensor noise should add variance"

    # Salt and pepper pixels should be present
    salt = np.sum(img_np == 255)
    pepper = np.sum(img_np == 0)
    assert salt + pepper > 0, "Salt and pepper noise should create extreme pixel values"


def test_data_augmenter_full():
    config = {
        "enabled": True,
        "flip_prob": 0.5,
        "rotation_range": [-30, 30],
        "scaling_range": [0.8, 1.2],
        "translation": [0.10, 0.10],
        "occlusion_prob": 0.5,
        "cutout_prob": 1.0,
        "cutout_size_ratio": 0.35,
        "intensity_jitter_prob": 1.0,
        "intensity_jitter_range": [0.6, 0.9],
        "contrast_jitter_range": [0.5, 0.8],
        "sensor_noise_prob": 1.0,
        "sensor_noise_sigma": [5.0, 10.0],
    }

    augmenter = DataAugmenter(config=config, is_training=True)
    img = Image.fromarray(np.ones((256, 256), dtype=np.uint8) * 128)

    # Sample 14 joints to match dataset layout
    joints = np.zeros((3, 14), dtype=np.float32)
    joints[0, :] = np.linspace(50, 200, 14)
    joints[1, :] = np.linspace(60, 210, 14)
    joints[2, :] = 2.0  # all visible

    img_aug, joints_aug = augmenter(img, joints, is_ir=True)

    # Check that augmentations ran and joints coordinates are present
    assert img_aug is not None
    assert joints_aug.shape == (3, 14)
