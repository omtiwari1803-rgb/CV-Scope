"""
CV-Scope Enhancement Module.

Implements classical image enhancement techniques:
- Global Histogram Equalization
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Spatial Convolution Sharpening (High-pass Laplacian & Unsharp Masking)
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np

from src.utils import validate_image, Timer, compute_image_statistics
from config import EnhancementConfig


def equalize_histogram(gray_img: np.ndarray) -> np.ndarray:
    """
    Perform global histogram equalization on a single-channel image.

    Linearizes the cumulative distribution function (CDF) of pixel intensities,
    maximizing global contrast across the full dynamic range [0, 255].

    Args:
        gray_img: Single-channel uint8 grayscale image.

    Returns:
        np.ndarray: Globally equalized grayscale image.
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("Histogram equalization requires a 2D single-channel grayscale image.")

    return cv2.equalizeHist(gray_img)


def apply_clahe(
    gray_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Overcomes limitations of global equalization by computing localized histograms
    over contextual tiles, clipping contrast amplification at clip_limit to prevent
    noise over-amplification, and bilinearly interpolating tile boundaries.

    Args:
        gray_img: Single-channel uint8 grayscale image.
        clip_limit: Threshold for contrast limiting (typical range: 1.5 - 4.0).
        tile_grid_size: Dimensions of the grid for contextual tile computation (e.g. (8, 8)).

    Returns:
        np.ndarray: Locally contrast-enhanced grayscale image.
    """
    validate_image(gray_img)
    if len(gray_img.shape) != 2:
        raise ValueError("CLAHE requires a 2D single-channel grayscale image.")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")
    if tile_grid_size[0] < 1 or tile_grid_size[1] < 1:
        raise ValueError(f"tile_grid_size dimensions must be >= 1, got {tile_grid_size}.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def sharpen_image(
    img: np.ndarray,
    strength: float = 1.0,
    method: str = "unsharp_mask",
    gaussian_ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0
) -> np.ndarray:
    """
    Enhance high-frequency edge details using spatial convolution.

    Methods:
    - 'unsharp_mask': Sharp = Image + strength * (Image - GaussianBlur(Image))
    - 'laplacian': 2D convolution with 3x3 discrete Laplacian high-pass filter.

    Args:
        img: Input image array.
        strength: Sharpening amplification multiplier (>= 0.0).
        method: Sharpening algorithm ('unsharp_mask' or 'laplacian').
        gaussian_ksize: Kernel dimensions for unsharp mask blur.
        sigma: Standard deviation for unsharp mask blur.

    Returns:
        np.ndarray: Sharpened uint8 image.
    """
    validate_image(img)
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}.")

    if strength == 0.0:
        return img.copy()

    if method == "unsharp_mask":
        blurred = cv2.GaussianBlur(img, gaussian_ksize, sigmaX=sigma, sigmaY=sigma)
        # Unsharp mask formula: High-frequency detail = img - blurred
        # Enhanced = img + strength * detail
        detail = cv2.subtract(img, blurred)
        sharpened = cv2.addWeighted(img, 1.0, detail, strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    elif method == "laplacian":
        # 3x3 Laplacian sharpening kernel with center weight
        # [ 0, -1,  0]
        # [-1,  4+strength, -1]
        # [ 0, -1,  0]
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1.0 + 4.0 * strength, -strength],
            [0, -strength, 0]
        ], dtype=np.float32)

        sharpened = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown sharpening method '{method}'. Supported: 'unsharp_mask', 'laplacian'.")


def run_enhancement(
    gray_img: np.ndarray,
    config: Optional[EnhancementConfig] = None
) -> Dict[str, Any]:
    """
    Execute classical image enhancement pipeline and calculate comparative metrics.

    Returns:
        dict:
        - 'original_gray': input grayscale image
        - 'equalized': global histogram equalized image
        - 'clahe': CLAHE enhanced image
        - 'sharpened': unsharp-mask sharpened image
        - 'metrics': statistical distributions before and after enhancement
        - 'elapsed_ms': total execution time in milliseconds
    """
    if config is None:
        config = EnhancementConfig()

    with Timer("Enhancement") as timer:
        eq = equalize_histogram(gray_img)
        clahe_out = apply_clahe(
            gray_img,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size
        )
        sharp = sharpen_image(
            clahe_out,
            strength=config.sharpen_strength,
            sigma=config.unsharp_sigma
        )

        metrics = {
            "original": compute_image_statistics(gray_img),
            "equalized": compute_image_statistics(eq),
            "clahe": compute_image_statistics(clahe_out),
            "sharpened": compute_image_statistics(sharp)
        }

    return {
        "original_gray": gray_img,
        "equalized": eq,
        "clahe": clahe_out,
        "sharpened": sharp,
        "metrics": metrics,
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
