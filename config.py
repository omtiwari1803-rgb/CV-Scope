"""
CV-Scope Configuration Module.

Provides typed configuration dataclasses with sensible defaults for all
classical Computer Vision algorithms across the CV-Scope pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing operations."""
    target_size: Optional[Tuple[int, int]] = None  # (width, height)
    scale_factor: Optional[float] = None
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    median_kernel_size: int = 5
    normalize_min: float = 0.0
    normalize_max: float = 255.0


@dataclass
class EnhancementConfig:
    """Configuration for contrast and frequency enhancement."""
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    sharpen_strength: float = 1.0
    unsharp_sigma: float = 1.0


@dataclass
class EdgeSegmentationConfig:
    """Configuration for edge detection and region segmentation."""
    sobel_kernel_size: int = 3
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    threshold_method: str = "otsu"  # "otsu" or "adaptive"


@dataclass
class FeatureConfig:
    """Configuration for SIFT feature detection and matching."""
    n_features: int = 2000
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    matcher_type: str = "flann"  # "flann" or "bf"
    ratio_threshold: float = 0.75  # Lowe's ratio test


@dataclass
class GeometryConfig:
    """Configuration for homography and projective transformations."""
    ransac_reproj_threshold: float = 3.0  # pixels
    max_ransac_iters: int = 2000
    confidence: float = 0.99
    min_inliers_required: int = 10


@dataclass
class EpipolarConfig:
    """Configuration for fundamental matrix and epipolar geometry."""
    ransac_threshold: float = 1.0  # distance to epipolar line in pixels
    confidence: float = 0.99
    max_epipolar_lines_to_draw: int = 20
    min_correspondences: int = 8


@dataclass
class StereoConfig:
    """Configuration for stereo correspondence and depth estimation."""
    algorithm: str = "sgbm"  # "sgbm" or "bm"
    min_disparity: int = 0
    num_disparities: int = 64  # Must be divisible by 16
    block_size: int = 7       # Odd number, typically 3 to 11
    # StereoSGBM specific smoothness parameters
    p1: Optional[int] = None  # Penalty on small disparity changes (8 * num_channels * block_size^2)
    p2: Optional[int] = None  # Penalty on large disparity changes (32 * num_channels * block_size^2)
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    mode: int = 0             # cv2.STEREO_SGBM_MODE_SGBM
    # Camera geometry parameters
    focal_length: float = 800.0  # Camera focal length in pixels
    baseline: float = 0.1        # Stereo baseline in meters


@dataclass
class PipelineConfig:
    """Global configuration orchestrating all modules in CV-Scope."""
    output_dir: str = "outputs"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    edge_segmentation: EdgeSegmentationConfig = field(default_factory=EdgeSegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    epipolar: EpipolarConfig = field(default_factory=EpipolarConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
