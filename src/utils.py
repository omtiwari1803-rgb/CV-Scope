"""
CV-Scope Core Utilities.

Provides robust image I/O, validation, directory management, performance timing,
and numerical statistics for image processing pipelines.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import cv2
import numpy as np


class Timer:
    """High-precision context manager and utility for execution timing."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that the target directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_image(
    img: Optional[np.ndarray],
    min_dim: Tuple[int, int] = (10, 10),
    allow_grayscale: bool = True
) -> None:
    """
    Validate that an image array is well-formed, non-empty, and meets dimension criteria.

    Raises:
        ValueError: If image is None, empty, or has dimensions smaller than min_dim.
    """
    if img is None:
        raise ValueError("Image array is None. Failed to load or decode image.")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(img).__name__}.")
    if img.size == 0:
        raise ValueError("Image array is empty (0 pixels).")
    if len(img.shape) < 2:
        raise ValueError(f"Image array must have at least 2 dimensions, got shape {img.shape}.")

    h, w = img.shape[:2]
    min_w, min_h = min_dim
    if h < min_h or w < min_w:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are smaller than minimum allowed ({min_w}x{min_h})."
        )

    if not allow_grayscale and len(img.shape) == 2:
        raise ValueError("Expected multi-channel color image, but received single-channel grayscale.")


def load_image(
    path: Union[str, Path],
    as_gray: bool = False
) -> np.ndarray:
    """
    Load an image from disk with validation and color space handling.

    Color images are converted from OpenCV's default BGR to RGB.

    Args:
        path: Path to image file.
        as_gray: If True, loads directly as single-channel grayscale.

    Returns:
        np.ndarray: Loaded image array in RGB (uint8) or Grayscale (uint8).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is not a valid or readable image.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path.resolve()}")
    if not file_path.is_file():
        raise ValueError(f"Specified path is not a file: {file_path.resolve()}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {file_path.resolve()}")

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(file_path), flag)

    if img is None:
        raise ValueError(
            f"Unable to decode image from '{file_path}'. "
            "File may be corrupted or in an unsupported format."
        )

    if not as_gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    validate_image(img)
    return img


def save_image(
    path: Union[str, Path],
    img: np.ndarray
) -> str:
    """
    Save an image array to disk, handling color space conversion (RGB to BGR).

    Args:
        path: Target file path.
        img: Image numpy array.

    Returns:
        str: Absolute path to the saved file.

    Raises:
        ValueError: If image is invalid or cv2.imwrite fails.
    """
    validate_image(img)
    out_path = Path(path)
    ensure_dir(out_path.parent)

    save_arr = img
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert RGB to BGR for OpenCV write
        save_arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(out_path), save_arr)
    if not success:
        raise IOError(f"Failed to write image to disk: {out_path.resolve()}")

    return str(out_path.resolve())


def compute_image_statistics(img: np.ndarray) -> Dict[str, Any]:
    """
    Compute key statistical metrics of an image channel or grayscale representation.

    Returns:
        dict: min, max, mean, std, dynamic_range, and contrast_ratio.
    """
    validate_image(img)
    arr = img.astype(np.float64)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    contrast_ratio = float(std_val / (mean_val + 1e-6))

    return {
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "dynamic_range": round(max_val - min_val, 2),
        "contrast_ratio": round(contrast_ratio, 4)
    }
