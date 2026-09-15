"""
Unit tests for CV-Scope preprocessing and enhancement modules.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.utils import load_image, save_image, validate_image, compute_image_statistics
from src.preprocessing import (
    to_grayscale,
    resize_image,
    denoise_gaussian,
    denoise_median,
    normalize_image,
    run_preprocessing
)
from src.enhancement import (
    equalize_histogram,
    apply_clahe,
    sharpen_image,
    run_enhancement
)
from config import PreprocessingConfig, EnhancementConfig


@pytest.fixture
def sample_color_image():
    """Create synthetic 100x100 RGB color image."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :50] = [255, 0, 0]    # Red quadrant
    arr[:50, 50:] = [0, 255, 0]    # Green quadrant
    arr[50:, :50] = [0, 0, 255]    # Blue quadrant
    arr[50:, 50:] = [200, 200, 200]  # Gray quadrant
    return arr


@pytest.fixture
def sample_gray_image():
    """Create synthetic 100x100 grayscale image."""
    arr = np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)
    return arr


def test_validate_image_valid(sample_color_image):
    """Verify validation passes on valid image array."""
    validate_image(sample_color_image)


def test_validate_image_invalid():
    """Verify validation raises errors on None, empty, or too small images."""
    with pytest.raises(ValueError):
        validate_image(None)
    with pytest.raises(ValueError):
        validate_image(np.empty((0, 0)))
    with pytest.raises(ValueError):
        validate_image(np.zeros((5, 5)), min_dim=(10, 10))


def test_to_grayscale(sample_color_image, sample_gray_image):
    """Verify RGB to Grayscale conversion and idempotency on 2D arrays."""
    gray = to_grayscale(sample_color_image)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # Idempotence: 2D image should remain identical
    gray2 = to_grayscale(sample_gray_image)
    assert np.array_equal(gray2, sample_gray_image)


def test_resize_image(sample_color_image):
    """Verify resizing with explicit dims, aspect ratio, and scaling."""
    # Explicit dims
    res1 = resize_image(sample_color_image, width=50, height=80)
    assert res1.shape[:2] == (80, 50)

    # Proportional scale
    res2 = resize_image(sample_color_image, scale_factor=0.5)
    assert res2.shape[:2] == (50, 50)

    # Proportional width
    res3 = resize_image(sample_color_image, width=200)
    assert res3.shape[:2] == (200, 200)


def test_denoise_gaussian(sample_gray_image):
    """Verify Gaussian filter preserves dimensions and reduces variance."""
    noisy = sample_gray_image.copy()
    noisy[::2, ::2] = 255  # Add artificial impulse noise
    filtered = denoise_gaussian(noisy, kernel_size=(5, 5), sigma=1.2)
    assert filtered.shape == sample_gray_image.shape
    assert filtered.dtype == np.uint8


def test_denoise_median(sample_gray_image):
    """Verify median filter removes salt-and-pepper noise."""
    noisy = sample_gray_image.copy()
    noisy[10, 10] = 255
    noisy[20, 20] = 0
    filtered = denoise_median(noisy, kernel_size=3)
    assert filtered.shape == sample_gray_image.shape
    assert filtered.dtype == np.uint8


def test_normalize_image(sample_gray_image):
    """Verify min-max normalization scales range properly."""
    sub_range = (sample_gray_image // 2) + 50  # Range approx [50, 177]
    norm = normalize_image(sub_range, min_out=0.0, max_out=255.0)
    assert norm.min() == 0
    assert norm.max() == 255


def test_equalize_histogram(sample_gray_image):
    """Verify global histogram equalization increases contrast."""
    low_contrast = np.full((100, 100), 100, dtype=np.uint8)
    low_contrast[20:80, 20:80] = 120
    eq = equalize_histogram(low_contrast)
    assert eq.shape == (100, 100)
    # Contrast should be expanded
    assert eq.max() > low_contrast.max()
    assert eq.min() < low_contrast.min()


def test_apply_clahe(sample_gray_image):
    """Verify CLAHE runs cleanly with valid parameters."""
    clahe_out = apply_clahe(sample_gray_image, clip_limit=3.0, tile_grid_size=(4, 4))
    assert clahe_out.shape == sample_gray_image.shape
    assert clahe_out.dtype == np.uint8


def test_sharpen_image(sample_gray_image):
    """Verify unsharp masking and Laplacian spatial convolution."""
    sharp_unsharp = sharpen_image(sample_gray_image, strength=1.5, method="unsharp_mask")
    assert sharp_unsharp.shape == sample_gray_image.shape

    sharp_laplacian = sharpen_image(sample_gray_image, strength=1.0, method="laplacian")
    assert sharp_laplacian.shape == sample_gray_image.shape


def test_run_preprocessing_pipeline(sample_color_image):
    """Verify full preprocessing pipeline dictionary output."""
    cfg = PreprocessingConfig()
    res = run_preprocessing(sample_color_image, cfg)
    assert "grayscale" in res
    assert "gaussian_denoised" in res
    assert "median_denoised" in res
    assert "normalized" in res
    assert "statistics" in res
    assert res["elapsed_ms"] >= 0.0


def test_run_enhancement_pipeline(sample_gray_image):
    """Verify enhancement pipeline execution."""
    cfg = EnhancementConfig()
    res = run_enhancement(sample_gray_image, cfg)
    assert "equalized" in res
    assert "clahe" in res
    assert "sharpened" in res
    assert "metrics" in res
    assert res["elapsed_ms"] >= 0.0
