"""
CV-Scope Image Geometry Module.

Implements planar projective transformations, homography estimation with RANSAC,
perspective warping, and geometric registration error analysis.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np

from src.utils import validate_image, Timer
from config import GeometryConfig


def compute_reprojection_error(
    pts1: np.ndarray,
    pts2: np.ndarray,
    H: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Calculate Euclidean reprojection error between mapped points H*x1 and observed points x2.

    Reprojection error: e_i = || x2_i - (H * x1_i)_normalized ||_2

    Args:
        pts1: (N, 2) source points.
        pts2: (N, 2) destination points.
        H: (3, 3) projective homography matrix.
        inlier_mask: (N,) binary inlier mask.

    Returns:
        Tuple[float, np.ndarray]:
        - mean_inlier_error: average reprojection error across inliers in pixels
        - point_errors: (N,) array of individual point errors
    """
    if len(pts1) == 0:
        return 0.0, np.array([], dtype=np.float32)

    # Convert pts1 to homogeneous coordinates (N, 3)
    n = pts1.shape[0]
    homogeneous_pts1 = np.hstack([pts1, np.ones((n, 1), dtype=np.float32)])

    # Map through homography: (3, 3) x (3, N) -> (3, N) -> transpose to (N, 3)
    projected = (H @ homogeneous_pts1.T).T

    # Normalize by z-coordinate
    eps = 1e-7
    z = projected[:, 2:3]
    # Avoid division by zero
    z = np.where(np.abs(z) < eps, np.sign(z + eps) * eps, z)
    projected_2d = projected[:, :2] / z

    # Calculate Euclidean distance
    errors = np.linalg.norm(projected_2d - pts2, axis=1)

    if inlier_mask is not None:
        mask_flat = inlier_mask.ravel().astype(bool)
        if np.any(mask_flat):
            mean_error = float(np.mean(errors[mask_flat]))
        else:
            mean_error = float(np.mean(errors))
    else:
        mean_error = float(np.mean(errors))

    return round(mean_error, 3), errors


def estimate_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    config: Optional[GeometryConfig] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Estimate 3x3 projective homography matrix using RANSAC.

    Homography maps points between two planar perspectives: x2 ~ H * x1.
    RANSAC iteratively selects 4-point minimal subsets, computes H using Direct Linear
    Transformation (DLT), and counts inliers within reprojection threshold to eliminate outliers.

    Args:
        pts1: (N, 2) source points.
        pts2: (N, 2) destination points.
        config: Geometry configuration parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
        - H: (3, 3) estimated homography matrix
        - inlier_mask: (N, 1) uint8 binary mask (1 for inlier, 0 for outlier)
        - inlier_ratio: fraction of matches that are inliers
        - mean_reproj_error: average reprojection error of inliers in pixels

    Raises:
        ValueError: If fewer than 4 corresponding points are provided.
        RuntimeError: If RANSAC fails to find a valid transformation.
    """
    if config is None:
        config = GeometryConfig()

    if len(pts1) < 4 or len(pts2) < 4:
        raise ValueError(
            f"Homography estimation requires at least 4 point correspondences. "
            f"Provided only {len(pts1)} points."
        )

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=config.max_ransac_iters,
        confidence=config.confidence
    )

    if H is None or mask is None:
        raise RuntimeError(
            "RANSAC failed to estimate a valid homography. Points may be collinear "
            "or contain excessive outliers."
        )

    num_inliers = int(np.sum(mask))
    inlier_ratio = round(float(num_inliers) / float(len(pts1)), 4)

    mean_err, _ = compute_reprojection_error(pts1, pts2, H, mask)

    return H, mask, inlier_ratio, mean_err


def warp_perspective(
    img: np.ndarray,
    H: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Warp source image using 3x3 homography matrix into target reference coordinate frame.

    Args:
        img: Source image (RGB or Grayscale).
        H: (3, 3) homography transformation matrix.
        target_shape: (width, height) in pixels.

    Returns:
        np.ndarray: Warped image.
    """
    validate_image(img)
    target_w, target_h = target_shape
    warped = cv2.warpPerspective(img, H, (target_w, target_h), flags=cv2.INTER_LINEAR)
    return warped


def create_alignment_overlay(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create transparent alpha-blended overlay between reference image and warped image.

    Green/Magenta or direct blending allows instant visual evaluation of geometric registration.

    Args:
        img_ref: Target reference image (RGB).
        img_warped: Warped source image in target coordinate space.
        alpha: Blending weight for reference image (0.0 to 1.0).

    Returns:
        np.ndarray: Blended RGB composite.
    """
    validate_image(img_ref)
    validate_image(img_warped)

    if img_ref.shape[:2] != img_warped.shape[:2]:
        raise ValueError("Reference and warped images must have identical spatial dimensions.")

    ref_rgb = img_ref if len(img_ref.shape) == 3 else cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB)
    warp_rgb = img_warped if len(img_warped.shape) == 3 else cv2.cvtColor(img_warped, cv2.COLOR_GRAY2RGB)

    blended = cv2.addWeighted(ref_rgb, alpha, warp_rgb, 1.0 - alpha, 0)
    return blended


def run_geometry_analysis(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    config: Optional[GeometryConfig] = None
) -> Dict[str, Any]:
    """
    Execute homography estimation, RANSAC inlier filtering, and perspective image alignment.

    Returns:
        dict:
        - 'homography_matrix': 3x3 homography matrix
        - 'inlier_mask': (N,) binary inlier flags
        - 'num_inliers': count of verified geometric inliers
        - 'inlier_ratio': ratio of inliers to total matches
        - 'reprojection_error': mean reprojection error in pixels
        - 'warped_image': image 1 warped into image 2 coordinates
        - 'alignment_overlay': alpha blended overlay image
        - 'elapsed_ms': execution time
    """
    if config is None:
        config = GeometryConfig()

    with Timer("Geometry & Homography") as timer:
        H, mask, inlier_ratio, reproj_err = estimate_homography(pts1, pts2, config)

        h2, w2 = img2.shape[:2]
        warped = warp_perspective(img1, H, (w2, h2))
        overlay = create_alignment_overlay(img2, warped)

    return {
        "homography_matrix": H.tolist(),
        "inlier_mask": mask.ravel().tolist(),
        "num_inliers": int(np.sum(mask)),
        "inlier_ratio": inlier_ratio,
        "reprojection_error": reproj_err,
        "warped_image": warped,
        "alignment_overlay": overlay,
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
