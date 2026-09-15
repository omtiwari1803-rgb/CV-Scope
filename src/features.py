"""
CV-Scope Feature Detection and Matching Module.

Implements classical scale- and rotation-invariant feature analysis using SIFT
(Scale-Invariant Feature Transform) and Lowe's ratio-tested descriptor matching.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from src.utils import validate_image, Timer
from config import FeatureConfig


def create_sift_detector(config: Optional[FeatureConfig] = None) -> cv2.SIFT:
    """
    Initialize OpenCV SIFT feature detector with configured parameters.

    SIFT algorithm principles:
    1. Scale-space extrema detection using Difference of Gaussians (DoG).
    2. Sub-pixel keypoint localization and removal of low-contrast/edge points.
    3. Orientation assignment based on local gradient directions.
    4. 128-dimensional descriptor vector computation from 4x4 subregions of 8-bin histograms.
    """
    if config is None:
        config = FeatureConfig()

    return cv2.SIFT_create(
        nfeatures=config.n_features,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma
    )


def detect_sift_features(
    gray_img: np.ndarray,
    config: Optional[FeatureConfig] = None
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Detect SIFT keypoints and compute their 128-dimensional descriptors.

    Args:
        gray_img: Single-channel uint8 grayscale image.
        config: Feature configuration parameters.

    Returns:
        Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        - keypoints: list of OpenCV KeyPoint objects
        - descriptors: (N, 128) float32 array or None if no features found
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("SIFT feature detection requires single-channel grayscale image.")

    detector = create_sift_detector(config)
    keypoints, descriptors = detector.detectAndCompute(gray_img, None)

    if descriptors is None or len(keypoints) == 0:
        return [], None

    return list(keypoints), descriptors


def match_features(
    desc1: Optional[np.ndarray],
    desc2: Optional[np.ndarray],
    matcher_type: str = "flann",
    ratio_threshold: float = 0.75
) -> Tuple[List[cv2.DMatch], List[List[cv2.DMatch]]]:
    """
    Match feature descriptors between two images and filter via Lowe's ratio test.

    Lowe's Ratio Test:
    Given two nearest neighbors m (closest) and n (second closest), a match is valid
    if dist(m) < ratio_threshold * dist(n). This eliminates false matches from repetitive
    textures or non-distinctive background structures.

    Args:
        desc1: Feature descriptors for image 1 (N1, 128).
        desc2: Feature descriptors for image 2 (N2, 128).
        matcher_type: 'flann' (Fast Library for Approximate Nearest Neighbors) or 'bf' (Brute-Force).
        ratio_threshold: Maximum allowable ratio dist(1st)/dist(2nd) (typically 0.7 - 0.8).

    Returns:
        Tuple[List[cv2.DMatch], List[List[cv2.DMatch]]]:
        - good_matches: 1D list of DMatch objects passing the ratio test
        - raw_knn_matches: raw 2-NN matches from matcher
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], []

    if matcher_type.lower() == "flann":
        # FLANN Kd-Tree index parameters for SIFT (floating-point descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif matcher_type.lower() == "bf":
        # Brute-Force Matcher with L2 distance norm
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        raise ValueError(f"Unknown matcher type '{matcher_type}'. Supported: 'flann', 'bf'.")

    # Query 2 nearest neighbors for each descriptor
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches: List[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches, knn_matches


def extract_matched_points(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matching (x, y) 2D coordinate pairs from keypoint lists and match objects.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - pts1: (M, 2) float32 coordinates from image 1
        - pts2: (M, 2) float32 coordinates from image 2
    """
    if not matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def render_keypoints(
    img: np.ndarray,
    keypoints: List[cv2.KeyPoint]
) -> np.ndarray:
    """
    Render keypoints with scale and orientation circles onto image.

    Returns:
        np.ndarray: RGB image with drawn keypoints.
    """
    validate_image(img)
    rendered = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 128)
    )
    return rendered


def render_matches(
    img1: np.ndarray,
    kps1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: Optional[np.ndarray] = None,
    max_matches_to_draw: int = 50
) -> np.ndarray:
    """
    Render side-by-side match connections between image pair.

    Args:
        img1: Left/first image.
        img2: Right/second image.
        kps1: Keypoints from image 1.
        kps2: Keypoints from image 2.
        matches: List of DMatch objects.
        inlier_mask: Optional binary mask (N,) or (N, 1) indicating inliers vs outliers.
        max_matches_to_draw: Subsampling limit for visual clarity.

    Returns:
        np.ndarray: RGB image visualizing correspondences.
    """
    validate_image(img1)
    validate_image(img2)

    draw_matches = matches[:max_matches_to_draw]
    matches_mask = None
    if inlier_mask is not None:
        flat_mask = inlier_mask.ravel().tolist()[:max_matches_to_draw]
        matches_mask = flat_mask

    draw_params = dict(
        matchColor=(0, 255, 0),       # Green for good matches / inliers
        singlePointColor=(255, 0, 0), # Red for isolated keypoints
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    rendered = cv2.drawMatches(
        img1, kps1, img2, kps2, draw_matches, None, **draw_params
    )
    return rendered


def run_feature_analysis(
    img1: np.ndarray,
    img2: Optional[np.ndarray] = None,
    config: Optional[FeatureConfig] = None
) -> Dict[str, Any]:
    """
    Execute SIFT detection on single image or pairwise matching across stereo/image pair.

    Returns:
        dict:
        - 'kps1': keypoint list for image 1
        - 'num_kps1': count of keypoints in image 1
        - 'kps1_visual': rendered keypoint image
        - If img2 is provided:
            - 'kps2', 'num_kps2', 'kps2_visual'
            - 'raw_matches_count': total KNN matches
            - 'good_matches_count': ratio-tested matches
            - 'good_matches': list of DMatch objects
            - 'match_ratio': good matches / min(num_kps1, num_kps2)
            - 'matches_visual': side-by-side rendered match visualization
            - 'pts1', 'pts2': extracted coordinate arrays
        - 'elapsed_ms': execution time
    """
    if config is None:
        config = FeatureConfig()

    with Timer("Feature Detection & Matching") as timer:
        # Convert image 1 to grayscale for SIFT
        gray1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        kps1, desc1 = detect_sift_features(gray1, config)
        kps1_vis = render_keypoints(img1, kps1)

        result: Dict[str, Any] = {
            "num_kps1": len(kps1),
            "kps1": kps1,
            "desc1": desc1,
            "kps1_visual": kps1_vis
        }

        if img2 is not None:
            gray2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            kps2, desc2 = detect_sift_features(gray2, config)
            kps2_vis = render_keypoints(img2, kps2)

            good_matches, knn_matches = match_features(
                desc1, desc2,
                matcher_type=config.matcher_type,
                ratio_threshold=config.ratio_threshold
            )

            pts1, pts2 = extract_matched_points(kps1, kps2, good_matches)
            matches_vis = render_matches(img1, kps1, img2, kps2, good_matches)

            min_kps = max(1, min(len(kps1), len(kps2)))
            match_ratio = round(len(good_matches) / min_kps, 4)

            result.update({
                "num_kps2": len(kps2),
                "kps2": kps2,
                "desc2": desc2,
                "kps2_visual": kps2_vis,
                "raw_matches_count": len(knn_matches),
                "good_matches_count": len(good_matches),
                "good_matches": good_matches,
                "match_ratio": match_ratio,
                "matches_visual": matches_vis,
                "pts1": pts1,
                "pts2": pts2
            })

    result["elapsed_ms"] = round(timer.elapsed_ms, 2)
    return result
