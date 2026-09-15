"""
CV-Scope Preprocessing Module.

Implements classical image preprocessing operations:
- Grayscale conversion
- Dimension scaling / aspect-ratio-preserving resizing
- Gaussian smoothing filter (linear filtering)
- Median denoising filter (non-linear salt-and-pepper reduction)
- Dynamic range normalization (min-max linear stretching)
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np

from src.utils import validate_image, Timer, compute_image_statistics
from config import PreprocessingConfig


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel 8-bit grayscale.

    If the image is already 2D (grayscale), returns a contiguous copy.
    Assumes 3-channel input is in RGB format.

    Args:
        img: Input image array (RGB or Grayscale).

    Returns:
        np.ndarray: Single-channel uint8 image.
    """
    validate_image(img)
    if len(img.shape) == 2:
        return np.ascontiguousarray(img, dtype=np.uint8)

    if img.shape[2] == 4:
        # RGBA -> Grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Standard RGB -> Grayscale using Rec.601 luma weights: Y = 0.299R + 0.587G + 0.114B
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def resize_image(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale_factor: Optional[float] = None,
    interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    Resize an image with explicit dimensions or proportional scaling.

    Args:
        img: Input image array.
        width: Target width in pixels.
        height: Target height in pixels.
        scale_factor: Proportional scaling multiplier.
        interpolation: OpenCV interpolation method (default: cv2.INTER_AREA for downsampling).

    Returns:
        np.ndarray: Resized image.
    """
    validate_image(img)
    h, w = img.shape[:2]

    if scale_factor is not None:
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
        new_w = max(1, int(round(w * scale_factor)))
        new_h = max(1, int(round(h * scale_factor)))
    elif width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive, got {width}x{height}.")
        new_w, new_h = width, height
    elif width is not None:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}.")
        new_w = width
        new_h = max(1, int(round(h * (width / w))))
    elif height is not None:
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}.")
        new_h = height
        new_w = max(1, int(round(w * (height / h))))
    else:
        return img.copy()

    # When upsampling, INTER_CUBIC or INTER_LINEAR is preferred
    interp = cv2.INTER_LINEAR if (new_w > w or new_h > h) else interpolation
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def denoise_gaussian(
    img: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing filter to suppress high-frequency additive noise.

    Args:
        img: Input image array.
        kernel_size: Tuple of positive odd integers (kw, kh).
        sigma: Standard deviation of the Gaussian kernel along both axes.

    Returns:
        np.ndarray: Gaussian smoothed image.
    """
    validate_image(img)
    kw, kh = kernel_size
    if kw % 2 == 0 or kh % 2 == 0 or kw < 1 or kh < 1:
        raise ValueError(f"Gaussian kernel sizes must be positive odd integers, got {kernel_size}.")

    return cv2.GaussianBlur(img, (kw, kh), sigmaX=sigma, sigmaY=sigma)


def denoise_median(
    img: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply non-linear median filtering for impulse and salt-and-pepper noise reduction.

    Median filtering replaces each pixel with the statistical median of neighboring
    pixels, effectively preserving sharp step edges while removing salt-and-pepper noise.

    Args:
        img: Input image array.
        kernel_size: Positive odd integer (e.g., 3, 5, 7).

    Returns:
        np.ndarray: Median filtered image.
    """
    validate_image(img)
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError(f"Median kernel size must be a positive odd integer, got {kernel_size}.")

    return cv2.medianBlur(img, kernel_size)


def normalize_image(
    img: np.ndarray,
    min_out: float = 0.0,
    max_out: float = 255.0,
    dtype: type = np.uint8
) -> np.ndarray:
    """
    Normalize image dynamic range using min-max linear stretching.

    Maps input pixel intensities [I_min, I_max] to target interval [min_out, max_out].

    Args:
        img: Input image array.
        min_out: Minimum output intensity.
        max_out: Maximum output intensity.
        dtype: Desired output numpy data type (default uint8).

    Returns:
        np.ndarray: Normalized image array.
    """
    validate_image(img)
    arr = img.astype(np.float64)
    i_min = np.min(arr)
    i_max = np.max(arr)

    if i_max - i_min < 1e-6:
        # Uniform image; map all values to mid-range
        stretched = np.full_like(arr, (min_out + max_out) / 2.0)
    else:
        stretched = (arr - i_min) / (i_max - i_min) * (max_out - min_out) + min_out

    return np.clip(stretched, min_out, max_out).astype(dtype)


def run_preprocessing(
    img: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> Dict[str, Any]:
    """
    Execute complete image preprocessing sequence with profiling.

    Returns dictionary containing:
    - 'original': validated source image
    - 'grayscale': single-channel luminance representation
    - 'gaussian_denoised': Gaussian filtered image
    - 'median_denoised': Median filtered image
    - 'normalized': min-max normalized image
    - 'statistics': intensity distribution statistics
    - 'elapsed_ms': execution time in milliseconds
    """
    if config is None:
        config = PreprocessingConfig()

    with Timer("Preprocessing") as timer:
        # 1. Dimension Resizing if configured
        processed = img
        if config.target_size or config.scale_factor:
            processed = resize_image(
                img,
                width=config.target_size[0] if config.target_size else None,
                height=config.target_size[1] if config.target_size else None,
                scale_factor=config.scale_factor
            )

        # 2. Luminance conversion
        gray = to_grayscale(processed)

        # 3. Gaussian smoothing
        gaussian = denoise_gaussian(
            gray,
            kernel_size=config.gaussian_kernel_size,
            sigma=config.gaussian_sigma
        )

        # 4. Median denoising
        median = denoise_median(gray, kernel_size=config.median_kernel_size)

        # 5. Normalization
        norm = normalize_image(
            gray,
            min_out=config.normalize_min,
            max_out=config.normalize_max
        )

        stats = compute_image_statistics(gray)

    return {
        "original": processed,
        "grayscale": gray,
        "gaussian_denoised": gaussian,
        "median_denoised": median,
        "normalized": norm,
        "statistics": stats,
        "dimensions": (processed.shape[1], processed.shape[0]),
        "elapsed_ms": round(timer.elapsed_ms, 2)
    }
