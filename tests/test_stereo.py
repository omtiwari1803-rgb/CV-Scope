"""
Unit tests for CV-Scope epipolar geometry, stereo correspondence, and depth reconstruction.
"""

import pytest
import numpy as np
import cv2

from src.epipolar import (
    estimate_fundamental_matrix,
    compute_epipolar_lines,
    draw_epipolar_geometry
)
from src.stereo import (
    validate_stereo_pair,
    compute_disparity,
    normalize_disparity_for_visualization,
    compute_depth_from_disparity,
    run_stereo_depth_pipeline
)
from config import StereoConfig, EpipolarConfig


@pytest.fixture
def stereo_test_pair():
    """Create a synthetic 100x120 stereo pair with a known horizontal shift."""
    h, w = 100, 120
    left = np.zeros((h, w), dtype=np.uint8)
    # Add textured pattern
    for y in range(10, 90, 10):
        for x in range(10, 110, 10):
            left[y:y+5, x:x+5] = 200

    # Right view has horizontal shift of d = 8 pixels
    M = np.float32([[1, 0, -8], [0, 1, 0]])
    right = cv2.warpAffine(left, M, (w, h))

    return left, right


def test_fundamental_matrix_insufficient_points():
    """Verify fundamental matrix raises ValueError when fewer than 8 correspondences provided."""
    pts1 = np.float32([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
    pts2 = np.float32([[12, 10], [22, 20], [32, 30], [42, 40], [52, 50]])

    with pytest.raises(ValueError, match="at least 8 point correspondences"):
        estimate_fundamental_matrix(pts1, pts2)


def test_epipolar_lines_computation():
    """Verify epipolar line shape and parameter computation."""
    pts = np.float32([[50, 50], [80, 80], [100, 60]])
    F = np.array([
        [0, 0, 0],
        [0, 0, -0.001],
        [0, 0.001, 0]
    ], dtype=np.float32)

    lines = compute_epipolar_lines(pts, which_image=1, F=F)
    assert lines.shape == (3, 3)


def test_validate_stereo_pair():
    """Verify stereo pair validation harmonizes mismatched image dimensions."""
    left = np.zeros((100, 150, 3), dtype=np.uint8)
    right = np.zeros((100, 140, 3), dtype=np.uint8)

    left_v, right_v = validate_stereo_pair(left, right)
    assert left_v.shape == right_v.shape
    assert right_v.shape == (100, 150, 3)


def test_compute_disparity_sgbm(stereo_test_pair):
    """Verify StereoSGBM computes non-empty disparity map."""
    left, right = stereo_test_pair
    cfg = StereoConfig(algorithm="sgbm", num_disparities=32, block_size=5)

    disp_float, raw_disp = compute_disparity(left, right, config=cfg)
    assert disp_float.shape == left.shape
    assert disp_float.dtype == np.float32


def test_compute_depth_formula():
    """
    Verify exact depth formula Z = (f * B) / d calculation:
    For f = 800 px, B = 0.1 m, d = 16.0 px:
    Z = (800 * 0.1) / 16.0 = 80 / 16 = 5.0 meters.
    """
    disp = np.array([
        [16.0, 32.0],
        [8.0, 0.0]  # 0.0 is invalid disparity
    ], dtype=np.float32)

    focal = 800.0
    baseline = 0.1

    depth_map, color_rgb, stats = compute_depth_from_disparity(
        disp, focal_length=focal, baseline=baseline, min_disp=0.5
    )

    assert depth_map.shape == (2, 2)
    assert np.isclose(depth_map[0, 0], 5.0, atol=1e-3)   # 80 / 16 = 5.0m
    assert np.isclose(depth_map[0, 1], 2.5, atol=1e-3)   # 80 / 32 = 2.5m
    assert np.isclose(depth_map[1, 0], 10.0, atol=1e-3)  # 80 / 8 = 10.0m
    assert depth_map[1, 1] == 0.0                        # Invalid disparity -> depth 0.0
    assert stats["valid_pixel_count"] == 3


def test_normalize_disparity_for_visualization():
    """Verify disparity normalization handles valid range and masks invalid pixels."""
    disp = np.array([
        [10.0, 20.0],
        [-1.0, 0.0]
    ], dtype=np.float32)

    gray_norm, color_rgb = normalize_disparity_for_visualization(disp, min_disp=0.0)

    assert gray_norm.shape == (2, 2)
    assert gray_norm[0, 0] == 0    # Min valid
    assert gray_norm[0, 1] == 255  # Max valid
    assert gray_norm[1, 0] == 0    # Masked
    assert gray_norm[1, 1] == 0    # Masked
    assert color_rgb.shape == (2, 2, 3)
