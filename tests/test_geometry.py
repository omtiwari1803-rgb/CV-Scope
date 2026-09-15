"""
Unit tests for CV-Scope image geometry, homography, and projective warping modules.
"""

import pytest
import numpy as np
import cv2

from src.geometry import (
    estimate_homography,
    compute_reprojection_error,
    warp_perspective,
    create_alignment_overlay,
    run_geometry_analysis
)
from config import GeometryConfig


@pytest.fixture
def synthetic_homography_points():
    """Generate known 2D points and mapped points under exact homography."""
    pts1 = np.float32([
        [50, 50],
        [200, 50],
        [200, 200],
        [50, 200],
        [100, 100],
        [150, 150],
        [75, 175],
        [175, 75]
    ])

    # Known homography: rotation by 15 deg + translation (10, 20) + small perspective
    H_true = np.array([
        [0.965, -0.258, 15.0],
        [0.258,  0.965, 20.0],
        [0.0001, 0.0002, 1.0]
    ], dtype=np.float32)

    # Project pts1 using H_true
    h_pts1 = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float32)])
    projected = (H_true @ h_pts1.T).T
    pts2 = projected[:, :2] / projected[:, 2:3]

    return pts1, pts2, H_true


def test_estimate_homography_clean(synthetic_homography_points):
    """Verify homography estimation recovers transformation with low error."""
    pts1, pts2, H_true = synthetic_homography_points
    cfg = GeometryConfig(ransac_reproj_threshold=2.0)

    H, mask, inlier_ratio, mean_err = estimate_homography(pts1, pts2, cfg)

    assert H.shape == (3, 3)
    assert inlier_ratio == 1.0
    assert mean_err < 1.0  # Sub-pixel accuracy on noise-free synthetic data


def test_estimate_homography_insufficient_points():
    """Verify homography raises ValueError when fewer than 4 points provided."""
    pts1 = np.float32([[10, 10], [20, 20], [30, 30]])
    pts2 = np.float32([[15, 15], [25, 25], [35, 35]])

    with pytest.raises(ValueError, match="at least 4 point correspondences"):
        estimate_homography(pts1, pts2)


def test_compute_reprojection_error():
    """Verify reprojection error is 0 for identity matrix mapping."""
    pts1 = np.float32([[10, 10], [50, 50], [100, 100]])
    pts2 = pts1.copy()
    H_ident = np.eye(3, dtype=np.float32)

    mean_err, point_errs = compute_reprojection_error(pts1, pts2, H_ident)
    assert mean_err == 0.0
    assert len(point_errs) == 3


def test_warp_perspective():
    """Verify projective warping produces output of specified dimensions."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    H = np.eye(3, dtype=np.float32)

    warped = warp_perspective(img, H, (150, 120))
    assert warped.shape == (120, 150, 3)


def test_create_alignment_overlay():
    """Verify alpha-blended overlay generation."""
    ref = np.full((100, 100, 3), 100, dtype=np.uint8)
    warp = np.full((100, 100, 3), 200, dtype=np.uint8)

    overlay = create_alignment_overlay(ref, warp, alpha=0.5)
    assert overlay.shape == (100, 100, 3)
    assert np.allclose(overlay[0, 0], [150, 150, 150], atol=2)
