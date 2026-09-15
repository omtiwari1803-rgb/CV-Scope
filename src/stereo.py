"""
CV-Scope Stereo Depth Estimation Module.

Implements classical binocular stereopsis:
- Stereo image dimension validation and rectification
- Disparity estimation via Semi-Global Block Matching (StereoSGBM) and Block Matching (StereoBM)
- Disparity normalization and false-color mapping
- Metric and relative depth calculation using the triangulation formula: Z = (f * B) / d
- Statistical profiling of disparity and depth distributions
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np

from src.utils import validate_image, Timer
from src.preprocessing import to_grayscale
from config import StereoConfig


def validate_stereo_pair(
    img_left: np.ndarray,
    img_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and harmonize dimensions of stereo image pair.

    If dimensions differ slightly, resizes right image to match left image.

    Args:
        img_left: Left camera image array.
        img_right: Right camera image array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left and right images with matching dimensions.
    """
    validate_image(img_left)
    validate_image(img_right)

    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    if (h_l, w_l) != (h_r, w_r):
        img_right = cv2.resize(img_right, (w_l, h_l), interpolation=cv2.INTER_LINEAR)

    return img_left, img_right


def rectify_uncalibrated_stereo(
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hartley's uncalibrated stereo rectification transforms from Fundamental Matrix.

    Warps left and right images such that epipolar lines become horizontal and collinear,
    enabling standard 1D horizontal disparity matching.

    Args:
        img_left: Left image.
        img_right: Right image.
        pts_left: (N, 2) inlier coordinates in left image.
        pts_right: (N, 2) inlier coordinates in right image.
        F: (3, 3) Fundamental Matrix.

    Returns:
        Tuple: (rectified_left, rectified_right, H1, H2)
    """
    h, w = img_left.shape[:2]
    success, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_left.reshape(-1, 1, 2),
        pts_right.reshape(-1, 1, 2),
        F,
        (w, h)
    )

    if not success:
        # Fallback to unrectified images
        return img_left, img_right, np.eye(3), np.eye(3)

    rect_left = cv2.warpPerspective(img_left, H1, (w, h))
    rect_right = cv2.warpPerspective(img_right, H2, (w, h))
    return rect_left, rect_right, H1, H2


def compute_disparity(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    config: Optional[StereoConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense horizontal disparity map using classical stereo correspondence.

    Algorithms:
    - StereoSGBM (Semi-Global Block Matching, Hirschmüller 2008):
      Approximates 2D MRF energy minimization along 1D paths, balancing matching accuracy
      and smoothness constraints across edges.
    - StereoBM (Block Matching):
      Local block correlation using Sum of Absolute Differences (SAD).

    Note: OpenCV stereo matchers return 16-bit signed fixed-point disparity with 4 fractional bits.
    Actual sub-pixel disparity = raw_disparity / 16.0.

    Args:
        gray_left: Rectified single-channel left image (uint8).
        gray_right: Rectified single-channel right image (uint8).
        config: Stereo configuration parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - disp_float: Sub-pixel float32 disparity in pixels
        - disp_raw: Raw int16 disparity from OpenCV matcher
    """
    validate_image(gray_left)
    validate_image(gray_right)
    if len(gray_left.shape) != 2 or len(gray_right.shape) != 2:
        raise ValueError("Stereo disparity requires single-channel grayscale images.")

    if config is None:
        config = StereoConfig()

    num_disp = config.num_disparities
    # Ensure num_disparities is a multiple of 16 as required by OpenCV
    if num_disp % 16 != 0 or num_disp <= 0:
        num_disp = max(16, ((num_disp + 15) // 16) * 16)

    block_size = config.block_size
    if block_size % 2 == 0 or block_size < 3:
        block_size = max(3, block_size | 1)

    if config.algorithm.lower() == "bm":
        matcher = cv2.StereoBM_create(
            numDisparities=num_disp,
            blockSize=block_size
        )
        matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher.setPreFilterCap(31)
        matcher.setMinDisparity(config.min_disparity)
        matcher.setTextureThreshold(10)
        matcher.setUniquenessRatio(config.uniqueness_ratio)
        matcher.setSpeckleWindowSize(config.speckle_window_size)
        matcher.setSpeckleRange(config.speckle_range)

    elif config.algorithm.lower() == "sgbm":
        p1 = config.p1 if config.p1 is not None else 8 * 1 * (block_size ** 2)
        p2 = config.p2 if config.p2 is not None else 32 * 1 * (block_size ** 2)

        matcher = cv2.StereoSGBM_create(
            minDisparity=config.min_disparity,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=config.disp12_max_diff,
            uniquenessRatio=config.uniqueness_ratio,
            speckleWindowSize=config.speckle_window_size,
            speckleRange=config.speckle_range,
            mode=config.mode
        )
    else:
        raise ValueError(f"Unknown stereo algorithm '{config.algorithm}'. Supported: 'sgbm', 'bm'.")

    raw_disp = matcher.compute(gray_left, gray_right)
    # Convert fixed-point int16 to float32 subpixel disparity
    disp_float = raw_disp.astype(np.float32) / 16.0

    return disp_float, raw_disp


def normalize_disparity_for_visualization(
    disparity: np.ndarray,
    min_disp: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter invalid disparities and map valid range to 8-bit grayscale and false-color.

    Invalid disparity pixels (<= min_disp or non-finite) are masked out (black).

    Args:
        disparity: Sub-pixel float32 disparity map.
        min_disp: Lower threshold for valid disparity.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - norm_gray: uint8 grayscale disparity map [0, 255]
        - color_map: RGB false-color representation (Inferno colormap)
    """
    valid_mask = (disparity > min_disp) & np.isfinite(disparity)

    norm_gray = np.zeros_like(disparity, dtype=np.uint8)
    if np.any(valid_mask):
        valid_vals = disparity[valid_mask]
        d_min = np.min(valid_vals)
        d_max = np.max(valid_vals)
        if d_max > d_min:
            scaled = (valid_vals - d_min) / (d_max - d_min) * 255.0
            norm_gray[valid_mask] = scaled.astype(np.uint8)
        else:
            norm_gray[valid_mask] = 128

    # Generate false-color map with black for invalid regions
    color_bgr = cv2.applyColorMap(norm_gray, cv2.COLORMAP_INFERNO)
    color_bgr[~valid_mask] = [0, 0, 0]
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    return norm_gray, color_rgb


def compute_depth_from_disparity(
    disparity: np.ndarray,
    focal_length: float = 800.0,
    baseline: float = 0.1,
    min_disp: float = 0.5,
    max_depth_clip: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Reconstruct depth Z using standard stereo triangulation geometry:
        Z = (f * B) / d

    where:
    - Z = depth (distance from camera baseline plane along optical axis)
    - f = focal length in pixels
    - B = baseline in meters (or relative units)
    - d = horizontal disparity in pixels

    Depth Nature:
    - If f (pixels) and B (meters) are physically calibrated, Z represents metric depth in meters.
    - If uncalibrated, Z represents relative inverse-disparity scene geometry.

    Args:
        disparity: Sub-pixel float32 disparity map.
        focal_length: Focal length f.
        baseline: Baseline distance B.
        min_disp: Minimum positive disparity threshold to prevent division by zero/negative values.
        max_depth_clip: Upper threshold to truncate distant horizon outliers.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]:
        - depth_map: float32 depth array (0.0 for invalid/occluded pixels)
        - depth_colormap: RGB false-color representation (Viridis colormap)
        - statistics: distribution metrics (min, max, mean, median, valid_ratio)
    """
    if focal_length <= 0:
        raise ValueError(f"Focal length must be positive, got {focal_length}.")
    if baseline <= 0:
        raise ValueError(f"Stereo baseline must be positive, got {baseline}.")

    valid_mask = (disparity >= min_disp) & np.isfinite(disparity)
    depth_map = np.zeros_like(disparity, dtype=np.float32)

    total_pixels = float(disparity.size)
    valid_count = int(np.count_nonzero(valid_mask))
    valid_ratio = round(valid_count / total_pixels, 4)

    if valid_count > 0:
        # Triangulation formula: Z = (f * B) / d
        depth_vals = (focal_length * baseline) / disparity[valid_mask]

        if max_depth_clip is None:
            # Robust auto-clip at 95th percentile to prevent infinite horizon scaling
            p95 = float(np.percentile(depth_vals, 95))
            clip_val = max(p95, float(np.median(depth_vals)) * 3.0)
        else:
            clip_val = max_depth_clip

        depth_clipped = np.clip(depth_vals, 0.0, clip_val)
        depth_map[valid_mask] = depth_clipped

        stats = {
            "valid_pixel_count": valid_count,
            "valid_ratio": valid_ratio,
            "min_depth": round(float(np.min(depth_vals)), 3),
            "max_depth": round(float(np.max(depth_clipped)), 3),
            "mean_depth": round(float(np.mean(depth_clipped)), 3),
            "median_depth": round(float(np.median(depth_vals)), 3),
            "depth_type": "metric (meters)" if baseline < 10 else "relative"
        }

        # Normalize depth map for false-color visualization (closer objects brighter)
        norm_depth = np.zeros_like(disparity, dtype=np.uint8)
        d_min = np.min(depth_clipped)
        d_max = np.max(depth_clipped)
        if d_max > d_min:
            # Invert so nearest objects have high intensity
            scaled = (1.0 - (depth_clipped - d_min) / (d_max - d_min)) * 255.0
            norm_depth[valid_mask] = scaled.astype(np.uint8)
        else:
            norm_depth[valid_mask] = 128

        color_bgr = cv2.applyColorMap(norm_depth, cv2.COLORMAP_VIRIDIS)
        color_bgr[~valid_mask] = [0, 0, 0]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    else:
        stats = {
            "valid_pixel_count": 0,
            "valid_ratio": 0.0,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "mean_depth": 0.0,
            "median_depth": 0.0,
            "depth_type": "insufficient_disparity"
        }
        color_rgb = np.zeros((disparity.shape[0], disparity.shape[1], 3), dtype=np.uint8)

    return depth_map, color_rgb, stats


def run_stereo_depth_pipeline(
    img_left: np.ndarray,
    img_right: np.ndarray,
    config: Optional[StereoConfig] = None
) -> Dict[str, Any]:
    """
    Execute end-to-end stereo matching and relative/metric depth computation.

    Returns:
        dict:
        - 'disparity': raw float32 disparity in pixels
        - 'disparity_normalized': uint8 normalized disparity
        - 'disparity_colored': RGB false-color disparity map
        - 'depth': float32 depth map
        - 'depth_colored': RGB false-color depth map
        - 'disparity_stats': disparity distribution statistics
        - 'depth_stats': depth distribution statistics
        - 'elapsed_ms': execution time in milliseconds
    """
    if config is None:
        config = StereoConfig()

    with Timer("Stereo Depth Analysis") as timer:
        # 1. Harmonize dimensions
        left, right = validate_stereo_pair(img_left, img_right)

        # 2. Grayscale conversion for stereo matching
        gray_l = to_grayscale(left)
        gray_r = to_grayscale(right)

        # 3. Disparity estimation
        disp_float, raw_disp = compute_disparity(gray_l, gray_r, config)

        # 4. Normalization and colorization
        disp_gray, disp_color = normalize_disparity_for_visualization(
            disp_float, min_disp=float(config.min_disparity)
        )

        valid_disp = disp_float[disp_float > config.min_disparity]
        disp_stats = {
            "algorithm": config.algorithm.upper(),
            "num_disparities": config.num_disparities,
            "block_size": config.block_size,
            "min_disparity": round(float(np.min(valid_disp)), 2) if len(valid_disp) > 0 else 0.0,
            "max_disparity": round(float(np.max(valid_disp)), 2) if len(valid_disp) > 0 else 0.0,
            "mean_disparity": round(float(np.mean(valid_disp)), 2) if len(valid_disp) > 0 else 0.0,
            "valid_percentage": round(float(len(valid_disp)) / float(disp_float.size) * 100.0, 2)
        }

        # 5. Triangulation Depth Recovery: Z = (f * B) / d
        depth_map, depth_color, depth_stats = compute_depth_from_disparity(
            disp_float,
            focal_length=config.focal_length,
            baseline=config.baseline,
            min_disp=max(0.5, float(config.min_disparity))
        )

    return {
        "disparity": disp_float,
        "disparity_normalized": disp_gray,
        "disparity_colored": disp_color,
        "depth": depth_map,
        "depth_colored": depth_color,
        "disparity_stats": disp_stats,
        "depth_stats": depth_stats,
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
