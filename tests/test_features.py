"""
Unit tests for CV-Scope SIFT feature detection and matching modules.
"""

import pytest
import numpy as np
import cv2

from src.features import (
    detect_sift_features,
    match_features,
    extract_matched_points,
    render_keypoints,
    render_matches,
    run_feature_analysis
)
from config import FeatureConfig


@pytest.fixture
def textured_pair():
    """Create a pair of textured images with known synthetic horizontal shift."""
    h, w = 200, 200
    img1 = np.zeros((h, w), dtype=np.uint8)
    # Add high-frequency checkerboard
    for y in range(20, 180, 20):
        for x in range(20, 180, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                img1[y:y+20, x:x+20] = 230
            else:
                img1[y:y+20, x:x+20] = 30

    # Image 2 is translated by dx=10, dy=0
    M = np.float32([[1, 0, 10], [0, 1, 0]])
    img2 = cv2.warpAffine(img1, M, (w, h))

    return img1, img2


def test_sift_detection_textured(textured_pair):
    """Verify SIFT detects keypoints and extracts 128-dimensional descriptors."""
    img1, _ = textured_pair
    kps, desc = detect_sift_features(img1)

    assert len(kps) > 0
    assert desc is not None
    assert desc.shape[0] == len(kps)
    assert desc.shape[1] == 128  # SIFT descriptor dimensionality
    assert desc.dtype == np.float32


def test_sift_detection_blank():
    """Verify blank uniform image gracefully returns empty keypoints and None descriptors."""
    blank = np.zeros((100, 100), dtype=np.uint8)
    kps, desc = detect_sift_features(blank)
    assert len(kps) == 0
    assert desc is None


def test_feature_matching_flann(textured_pair):
    """Verify FLANN matcher and Lowe's ratio test find valid correspondences."""
    img1, img2 = textured_pair
    kps1, desc1 = detect_sift_features(img1)
    kps2, desc2 = detect_sift_features(img2)

    good_matches, knn_matches = match_features(desc1, desc2, matcher_type="flann", ratio_threshold=0.8)
    assert len(good_matches) > 0
    assert len(good_matches) <= len(knn_matches)


def test_feature_matching_bf(textured_pair):
    """Verify Brute-Force matcher finds correspondences."""
    img1, img2 = textured_pair
    kps1, desc1 = detect_sift_features(img1)
    kps2, desc2 = detect_sift_features(img2)

    good_matches, _ = match_features(desc1, desc2, matcher_type="bf", ratio_threshold=0.8)
    assert len(good_matches) > 0


def test_extract_matched_points(textured_pair):
    """Verify extracted coordinate arrays have shape (N, 2) and float32 type."""
    img1, img2 = textured_pair
    kps1, desc1 = detect_sift_features(img1)
    kps2, desc2 = detect_sift_features(img2)
    good_matches, _ = match_features(desc1, desc2)

    pts1, pts2 = extract_matched_points(kps1, kps2, good_matches)
    assert pts1.shape == (len(good_matches), 2)
    assert pts2.shape == (len(good_matches), 2)
    assert pts1.dtype == np.float32


def test_run_feature_analysis_single(textured_pair):
    """Verify single-image analysis structure."""
    img1, _ = textured_pair
    res = run_feature_analysis(img1)
    assert "num_kps1" in res
    assert res["num_kps1"] > 0
    assert "kps1_visual" in res
    assert "good_matches" not in res


def test_run_feature_analysis_pair(textured_pair):
    """Verify pairwise analysis structure."""
    img1, img2 = textured_pair
    res = run_feature_analysis(img1, img2)
    assert "num_kps1" in res
    assert "num_kps2" in res
    assert "good_matches_count" in res
    assert "match_ratio" in res
    assert "matches_visual" in res
