"""
CV-Scope Visualization Module.

Generates publication-quality, headless-compatible visual artifacts for all
computer vision stages using Matplotlib (Agg backend) and OpenCV.
No GUI display windows are created.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import matplotlib
matplotlib.use("Agg")  # Strictly headless backend
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.utils import ensure_dir, save_image


def save_figure(fig: plt.Figure, output_path: Union[str, Path], dpi: int = 150) -> str:
    """Save a Matplotlib figure to disk and close it to release memory."""
    p = Path(output_path)
    ensure_dir(p.parent)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p.resolve())


def plot_enhancement_comparison(
    orig: np.ndarray,
    gray: np.ndarray,
    denoised: np.ndarray,
    clahe: np.ndarray,
    sharpened: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Generate a 2x3 comparative panel displaying preprocessing and enhancement steps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("CV-Scope: Preprocessing & Contrast Enhancement", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("1. Original RGB", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("2. Grayscale (Luminance)", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(denoised, cmap="gray")
    axes[0, 2].set_title("3. Gaussian Denoised", fontsize=12)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(clahe, cmap="gray")
    axes[1, 0].set_title("4. CLAHE Enhanced", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sharpened, cmap="gray")
    axes[1, 1].set_title("5. Unsharp Mask Sharpened", fontsize=12)
    axes[1, 1].axis("off")

    # Intensity histogram comparison
    ax_hist = axes[1, 2]
    ax_hist.hist(gray.ravel(), bins=64, range=(0, 256), color="gray", alpha=0.5, label="Original Gray")
    ax_hist.hist(clahe.ravel(), bins=64, range=(0, 256), color="teal", alpha=0.6, label="CLAHE")
    ax_hist.set_title("6. Intensity Histogram Comparison", fontsize=12)
    ax_hist.set_xlabel("Pixel Intensity [0-255]")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, output_path)


def plot_edge_segmentation_grid(
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    magnitude: np.ndarray,
    canny: np.ndarray,
    binary: np.ndarray,
    regions_color: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Generate a 2x3 panel illustrating edge derivatives, gradient field, and segmentation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("CV-Scope: Edge Detection & Spatial Region Segmentation", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(sobel_x, cmap="gray")
    axes[0, 0].set_title("Sobel X (Horizontal Gradient)", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sobel_y, cmap="gray")
    axes[0, 1].set_title("Sobel Y (Vertical Gradient)", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(magnitude, cmap="magma")
    axes[0, 2].set_title("Gradient Magnitude sqrt(Gx^2 + Gy^2)", fontsize=12)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(canny, cmap="gray")
    axes[1, 0].set_title("Canny Edges (Hysteresis)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(binary, cmap="gray")
    axes[1, 1].set_title("Otsu Binary Threshold", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(regions_color)
    axes[1, 2].set_title("Connected Component Regions", fontsize=12)
    axes[1, 2].axis("off")

    plt.tight_layout()
    return save_figure(fig, output_path)


def plot_geometry_alignment(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    overlay: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Plot reference image, homography-warped source image, and alpha-blended alignment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CV-Scope: Planar Homography & Projective Image Warping", fontsize=15, fontweight="bold")

    axes[0].imshow(img_ref)
    axes[0].set_title("Target Reference Image (View 2)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(img_warped)
    axes[1].set_title("Warped Source Image (H * View 1)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Alpha-Blended Alignment Registration", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    return save_figure(fig, output_path)


def plot_epipolar_lines(
    epipolar_composite: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """Save epipolar lines side-by-side composite."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(epipolar_composite)
    ax.set_title("CV-Scope: Epipolar Geometry & Conjugate Epipolar Lines (x2^T * F * x1 = 0)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return save_figure(fig, output_path)


def plot_disparity_and_depth(
    img_left: np.ndarray,
    disp_colored: np.ndarray,
    depth_colored: np.ndarray,
    disp_raw: np.ndarray,
    depth_raw: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Plot 2x2 comprehensive stereo analysis: Left View, Disparity Map, Depth Map, and Depth Histogram.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CV-Scope: Stereo Correspondence & Triangulation Depth (Z = f * B / d)", fontsize=15, fontweight="bold")

    axes[0, 0].imshow(img_left)
    axes[0, 0].set_title("Reference Left View", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(disp_colored)
    axes[0, 1].set_title("Disparity Map (StereoSGBM)", fontsize=12)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(depth_colored)
    axes[1, 0].set_title("Recovered Scene Depth Map", fontsize=12)
    axes[1, 0].axis("off")

    # Depth distribution histogram for valid pixels
    ax_hist = axes[1, 1]
    valid_depths = depth_raw[(depth_raw > 0) & np.isfinite(depth_raw)]
    if len(valid_depths) > 0:
        ax_hist.hist(valid_depths, bins=50, color="darkcyan", edgecolor="black", alpha=0.7)
        ax_hist.set_title(f"Valid Depth Distribution (N={len(valid_depths):,})", fontsize=12)
        ax_hist.set_xlabel("Depth Z (meters / relative units)")
        ax_hist.set_ylabel("Pixel Frequency")
        ax_hist.grid(True, alpha=0.3)
    else:
        ax_hist.text(0.5, 0.5, "No valid disparities detected", ha="center", va="center", fontsize=12)
        ax_hist.axis("off")

    plt.tight_layout()
    return save_figure(fig, output_path)
