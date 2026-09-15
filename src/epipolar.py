"""
CV-Scope Epipolar Geometry Module.

Implements two-view projective geometry, Fundamental Matrix estimation with RANSAC,
epipolar constraint verification (x2.T * F * x1 = 0), and epipolar line rendering.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from src.utils import validate_image, Timer
from config import EpipolarConfig


def estimate_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    config: Optional[EpipolarConfig] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate the 3x3 Fundamental Matrix (F) using RANSAC from point correspondences.

    Epipolar Constraint:
    For any pair of corresponding points x1 in image 1 and x2 in image 2 in homogeneous coordinates:
        x2^T * F * x1 = 0

    F encapsulates the intrinsic parameters of both cameras and their relative rotation (R)
    and translation (t), where F = K2^(-T) * [t]x * R * K1^(-1).

    Args:
        pts1: (N, 2) coordinates in image 1.
        pts2: (N, 2) coordinates in image 2.
        config: Epipolar configuration parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
        - F: (3, 3) fundamental matrix of rank 2
        - inlier_mask: (N, 1) uint8 binary mask of inlier correspondences
        - inlier_ratio: proportion of correspondences satisfying epipolar constraint

    Raises:
        ValueError: If fewer than 8 corresponding points are provided.
        RuntimeError: If RANSAC estimation fails.
    """
    if config is None:
        config = EpipolarConfig()

    if len(pts1) < 8 or len(pts2) < 8:
        raise ValueError(
            f"Fundamental matrix estimation requires at least 8 point correspondences (8-point algorithm). "
            f"Provided only {len(pts1)} points."
        )

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=config.ransac_threshold,
        confidence=config.confidence
    )

    if F is None or mask is None:
        raise RuntimeError(
            "Failed to estimate Fundamental Matrix. Correspondences may be degenerate, "
            "coplanar, or contain insufficient true matches."
        )

    # In case multiple 3x3 matrices are returned (7-point algorithm), select the first one
    if F.shape[0] > 3:
        F = F[:3, :3]

    num_inliers = int(np.sum(mask))
    inlier_ratio = round(float(num_inliers) / float(len(pts1)), 4)

    return F, mask, inlier_ratio


def compute_epipolar_lines(
    pts: np.ndarray,
    which_image: int,
    F: np.ndarray
) -> np.ndarray:
    """
    Calculate epipolar lines l = [a, b, c]^T such that a*x + b*y + c = 0.

    Args:
        pts: (N, 2) point coordinates.
        which_image: 1 if points are from image 1 (computes lines in image 2),
                     2 if points are from image 2 (computes lines in image 1).
        F: (3, 3) Fundamental matrix.

    Returns:
        np.ndarray: (N, 3) line parameters [a, b, c] for each point.
    """
    # cv2.computeCorrespondEpilines requires (N, 1, 2) shape
    reshaped_pts = pts.reshape(-1, 1, 2).astype(np.float32)
    lines = cv2.computeCorrespondEpilines(reshaped_pts, which_image, F)
    return lines.reshape(-1, 3)


def draw_epipolar_geometry(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_points: int = 15
) -> np.ndarray:
    """
    Render corresponding points and their associated epipolar lines on side-by-side images.

    Points in Image 1 map to Epipolar Lines in Image 2, and vice versa. Matching point-line
    pairs share identical distinct colors.

    Args:
        img1: Left image (RGB).
        img2: Right image (RGB).
        pts1: Coordinates in image 1.
        pts2: Coordinates in image 2.
        F: (3, 3) Fundamental Matrix.
        inlier_mask: Binary mask from RANSAC.
        max_points: Number of representative pairs to draw to avoid visual clutter.

    Returns:
        np.ndarray: Side-by-side RGB image visualizing epipolar geometry.
    """
    validate_image(img1)
    validate_image(img2)

    vis1 = img1.copy()
    vis2 = img2.copy()

    # Filter to inlier correspondences if mask provided
    if inlier_mask is not None:
        mask_flat = inlier_mask.ravel().astype(bool)
        p1 = pts1[mask_flat]
        p2 = pts2[mask_flat]
    else:
        p1 = pts1
        p2 = pts2

    if len(p1) == 0:
        # No points to draw; return side-by-side images
        return np.hstack([vis1, vis2])

    # Subsample points evenly
    indices = np.linspace(0, len(p1) - 1, min(max_points, len(p1)), dtype=int)
    sample_pts1 = p1[indices]
    sample_pts2 = p2[indices]

    # Compute epipolar lines
    # Lines in img1 corresponding to points in img2
    lines1 = compute_epipolar_lines(sample_pts2, 2, F)
    # Lines in img2 corresponding to points in img1
    lines2 = compute_epipolar_lines(sample_pts1, 1, F)

    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]

    # Generate distinct colors for each corresponding pair
    rng = np.random.default_rng(seed=42)
    colors = rng.integers(50, 255, size=(len(sample_pts1), 3)).tolist()

    for r1, r2, pt1, pt2, color in zip(lines1, lines2, sample_pts1, sample_pts2, colors):
        # Draw line in img1: r1 = [a, b, c] -> ax + by + c = 0 -> y = -(ax + c)/b
        if abs(r1[1]) > 1e-6:
            x0, y0 = 0, int(round(-r1[2] / r1[1]))
            x1, y1 = w1, int(round(-(r1[2] + r1[0] * w1) / r1[1]))
            cv2.line(vis1, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(vis1, (int(round(pt1[0])), int(round(pt1[1]))), 5, color, -1, cv2.LINE_AA)

        # Draw line in img2: r2 = [a, b, c]
        if abs(r2[1]) > 1e-6:
            x0, y0 = 0, int(round(-r2[2] / r2[1]))
            x1, y1 = w2, int(round(-(r2[2] + r2[0] * w2) / r2[1]))
            cv2.line(vis2, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(vis2, (int(round(pt2[0])), int(round(pt2[1]))), 5, color, -1, cv2.LINE_AA)

    # Concatenate side by side
    target_h = max(h1, h2)
    if h1 != target_h:
        vis1 = cv2.resize(vis1, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        vis2 = cv2.resize(vis2, (int(w2 * target_h / h2), target_h))

    composite = np.hstack([vis1, vis2])
    return composite


def run_epipolar_analysis(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    config: Optional[EpipolarConfig] = None
) -> Dict[str, Any]:
    """
    Execute full epipolar geometry analysis and generate side-by-side visualization.

    Returns:
        dict:
        - 'fundamental_matrix': (3, 3) matrix as nested list
        - 'num_inliers': count of verified epipolar inliers
        - 'inlier_ratio': ratio of inliers to total matches
        - 'inlier_mask': binary list of inlier indicators
        - 'epipolar_visual': composite image showing epipolar lines and matching points
        - 'elapsed_ms': execution time
    """
    if config is None:
        config = EpipolarConfig()

    with Timer("Epipolar Geometry") as timer:
        F, mask, inlier_ratio = estimate_fundamental_matrix(pts1, pts2, config)
        vis = draw_epipolar_geometry(
            img1, img2, pts1, pts2, F,
            inlier_mask=mask,
            max_points=config.max_epipolar_lines_to_draw
        )

    return {
        "fundamental_matrix": F.tolist(),
        "num_inliers": int(np.sum(mask)),
        "inlier_ratio": inlier_ratio,
        "inlier_mask": mask.ravel().tolist(),
        "epipolar_visual": vis,
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
