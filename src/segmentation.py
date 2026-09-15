"""
CV-Scope Edge and Segmentation Analysis Module.

Implements classical boundary extraction and spatial segmentation:
- Sobel directional spatial gradients (X and Y derivatives)
- Gradient magnitude and orientation estimation
- Canny edge detection (multi-stage non-maximum suppression and hysteresis)
- Otsu global thresholding & adaptive thresholding
- Morphological operations (opening, closing, gradient)
- Connected components spatial region labeling & morphological statistics
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np

from src.utils import validate_image, Timer
from config import EdgeSegmentationConfig


def compute_sobel(
    gray_img: np.ndarray,
    ksize: int = 3
) -> Dict[str, np.ndarray]:
    """
    Compute first-order spatial derivatives and gradient field using Sobel operator.

    Calculates:
    - Sobel X (horizontal gradient, dI/dx)
    - Sobel Y (vertical gradient, dI/dy)
    - Gradient Magnitude: G = sqrt(Gx^2 + Gy^2)
    - Gradient Orientation: theta = arctan2(Gy, Gx)

    Args:
        gray_img: Single-channel uint8 grayscale image.
        ksize: Aperture size for the Sobel filter (must be 1, 3, 5, or 7).

    Returns:
        dict:
        - 'sobel_x': float64 derivative along X
        - 'sobel_y': float64 derivative along Y
        - 'magnitude': float64 gradient magnitude
        - 'orientation': float64 orientation in radians [-pi, pi]
        - 'sobel_x_uint8': normalized uint8 representation
        - 'sobel_y_uint8': normalized uint8 representation
        - 'magnitude_uint8': normalized uint8 representation [0, 255]
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("Sobel gradient requires single-channel grayscale image.")
    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"Sobel ksize must be 1, 3, 5, or 7, got {ksize}.")

    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)

    mag, angle = cv2.cartToPolar(gx, gy)

    # Scale to 8-bit displayable formats
    # For directional gradients, map negative-to-positive range to [0, 255]
    abs_gx = cv2.convertScaleAbs(gx)
    abs_gy = cv2.convertScaleAbs(gy)
    mag_scaled = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return {
        "sobel_x": gx,
        "sobel_y": gy,
        "magnitude": mag,
        "orientation": angle,
        "sobel_x_uint8": abs_gx,
        "sobel_y_uint8": abs_gy,
        "magnitude_uint8": mag_scaled
    }


def detect_canny_edges(
    gray_img: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True
) -> np.ndarray:
    """
    Extract optimal step edges using Canny edge detector.

    Performs:
    1. Gaussian smoothing
    2. Gradient intensity computation
    3. Non-maximum suppression along gradient direction
    4. Hysteresis thresholding with dual thresholds

    Args:
        gray_img: Single-channel uint8 grayscale image.
        low_thresh: Lower hysteresis threshold for weak edges.
        high_thresh: Upper hysteresis threshold for strong edges.
        aperture_size: Sobel aperture size.
        l2_gradient: If True, uses Euclidean L2 norm sqrt(Gx^2 + Gy^2).

    Returns:
        np.ndarray: Binary edge map (0 or 255).
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("Canny edge detection requires single-channel grayscale image.")
    if low_thresh < 0 or high_thresh < 0:
        raise ValueError("Canny thresholds must be non-negative.")
    if low_thresh > high_thresh:
        # Auto-correct or warn
        low_thresh, high_thresh = high_thresh, low_thresh

    return cv2.Canny(
        gray_img,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=aperture_size,
        L2gradient=l2_gradient
    )


def apply_threshold(
    gray_img: np.ndarray,
    method: str = "otsu",
    block_size: int = 11,
    c_const: float = 2.0
) -> Tuple[np.ndarray, float]:
    """
    Segment foreground/background using global Otsu or adaptive Gaussian thresholding.

    Args:
        gray_img: Single-channel uint8 grayscale image.
        method: 'otsu' for intra-class variance minimization or 'adaptive' for localized thresholds.
        block_size: Neighborhood size for adaptive thresholding (must be odd).
        c_const: Constant subtracted from the mean or weighted mean in adaptive thresholding.

    Returns:
        Tuple[np.ndarray, float]: (binary_image, optimal_threshold_value)
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("Thresholding requires single-channel grayscale image.")

    if method.lower() == "otsu":
        # Otsu's bimodal thresholding computes optimal global threshold value T*
        thresh_val, binary = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary, float(thresh_val)
    elif method.lower() == "adaptive":
        if block_size % 2 == 0 or block_size < 3:
            raise ValueError(f"block_size must be odd and >= 3, got {block_size}.")
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_const
        )
        return binary, float(np.mean(binary))
    else:
        raise ValueError(f"Unknown thresholding method '{method}'. Supported: 'otsu', 'adaptive'.")


def morphological_cleanup(
    binary_img: np.ndarray,
    ksize: int = 3,
    iterations: int = 1,
    shape: int = cv2.MORPH_RECT
) -> Dict[str, np.ndarray]:
    """
    Refine binary segmentation using mathematical morphology operations.

    - Opening: Erosion followed by Dilation (eliminates minor isolated noise).
    - Closing: Dilation followed by Erosion (fills pinholes and connects contours).
    - Morphological Gradient: Dilation minus Erosion (highlights object boundary contours).

    Args:
        binary_img: Single-channel binary image (uint8).
        ksize: Size of the structuring element.
        iterations: Repetitions of morphological operations.
        shape: Structuring element geometry (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS).

    Returns:
        dict:
        - 'opened': cleaned binary map
        - 'closed': filled binary map
        - 'gradient': morphological perimeter gradient
    """
    validate_image(binary_img)
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    grad = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)

    return {
        "opened": opened,
        "closed": closed,
        "gradient": grad
    }


def segment_regions(
    binary_img: np.ndarray,
    min_area: int = 20
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Perform connected-component labeling and extract geometric region properties.

    Args:
        binary_img: Cleaned single-channel binary image.
        min_area: Minimum pixel area to retain region (filters small noise artifacts).

    Returns:
        Tuple[np.ndarray, List[Dict]]:
        - labeled_mask: 2D array of component IDs
        - region_stats: list of dicts containing area, centroid, and bounding boxes
    """
    validate_image(binary_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    regions = []
    for label_id in range(1, num_labels):  # Skip label 0 (background)
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        regions.append({
            "label": label_id,
            "area": area,
            "bbox": [x, y, w, h],
            "centroid": [round(float(cx), 2), round(float(cy), 2)]
        })

    return labels, regions


def create_color_label_map(labels: np.ndarray) -> np.ndarray:
    """Map integer region labels to a distinct pseudo-color RGB representation."""
    # Scale labels to [0, 255] and apply colormap
    norm_labels = cv2.normalize(labels.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_map = cv2.applyColorMap(norm_labels, cv2.COLORMAP_JET)
    # Zero out background (label 0)
    color_map[labels == 0] = [0, 0, 0]
    return cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)


def run_edge_segmentation(
    gray_img: np.ndarray,
    config: Optional[EdgeSegmentationConfig] = None
) -> Dict[str, Any]:
    """
    Execute full edge detection and segmentation suite with statistical summary.

    Returns:
        dict:
        - 'sobel': dictionary of Sobel derivatives and magnitude
        - 'canny': binary Canny edge map
        - 'binary_threshold': thresholded image
        - 'optimal_threshold': computed threshold value
        - 'morphology': dictionary of opened, closed, gradient maps
        - 'colored_regions': RGB colored segmentation map
        - 'edge_density': ratio of edge pixels to total image pixels
        - 'regions': list of segmented component statistics
        - 'num_regions': total valid detected components
        - 'elapsed_ms': execution time in milliseconds
    """
    if config is None:
        config = EdgeSegmentationConfig()

    with Timer("Edge & Segmentation") as timer:
        # 1. Sobel spatial gradients
        sobel_res = compute_sobel(gray_img, ksize=config.sobel_kernel_size)

        # 2. Canny edge detection
        canny = detect_canny_edges(
            gray_img,
            low_thresh=config.canny_low_threshold,
            high_thresh=config.canny_high_threshold
        )

        # 3. Edge density
        total_pixels = float(gray_img.shape[0] * gray_img.shape[1])
        edge_pixels = float(np.count_nonzero(canny))
        edge_density = round(edge_pixels / (total_pixels + 1e-6), 4)

        # 4. Binary thresholding
        binary, thresh_val = apply_threshold(gray_img, method=config.threshold_method)

        # 5. Morphological cleanup
        morph = morphological_cleanup(
            binary,
            ksize=config.morphology_kernel_size,
            iterations=config.morphology_iterations
        )

        # 6. Connected component region labeling
        labels, regions = segment_regions(morph["closed"])
        colored_labels = create_color_label_map(labels)

    return {
        "sobel": sobel_res,
        "canny": canny,
        "binary_threshold": binary,
        "optimal_threshold": round(thresh_val, 2),
        "morphology": morph,
        "colored_regions": colored_labels,
        "edge_density": edge_density,
        "regions": regions,
        "num_regions": len(regions),
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
