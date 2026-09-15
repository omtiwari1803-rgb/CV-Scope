"""
CV-Scope: Computer Vision Scene Geometry and Depth Analysis System.

Main Command-Line Interface (CLI) Controller.
Supports modular execution of classical Computer Vision algorithms and end-to-end pipelines.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PipelineConfig,
    PreprocessingConfig,
    EnhancementConfig,
    EdgeSegmentationConfig,
    FeatureConfig,
    GeometryConfig,
    EpipolarConfig,
    StereoConfig
)
from src.utils import load_image, save_image, ensure_dir, validate_image
from src.preprocessing import run_preprocessing, to_grayscale
from src.enhancement import run_enhancement
from src.segmentation import run_edge_segmentation
from src.features import run_feature_analysis
from src.geometry import run_geometry_analysis
from src.epipolar import run_epipolar_analysis
from src.stereo import run_stereo_depth_pipeline
from src.reporting import AnalysisReport
from src.visualization import (
    plot_enhancement_comparison,
    plot_edge_segmentation_grid,
    plot_geometry_alignment,
    plot_epipolar_lines,
    plot_disparity_and_depth
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with comprehensive help documentation."""
    parser = argparse.ArgumentParser(
        prog="cv-scope",
        description=(
            "CV-Scope: A classical Computer Vision Scene Geometry and Depth Analysis System.\n"
            "Demonstrates fundamental classical CV algorithms without deep learning models:\n"
            "Denoising, Enhancement, Gradients, Canny, Otsu, SIFT, RANSAC, Homography,\n"
            "Epipolar Geometry, StereoSGBM/BM, and Triangulation Depth Estimation (Z = f*B/d)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core Execution Modes
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["enhance", "edges", "features", "match", "geometry", "epipolar", "depth", "full"],
        help=(
            "Analysis mode to execute:\n"
            "  enhance   : Preprocessing, denoising, histogram equalization & CLAHE\n"
            "  edges     : Sobel gradients, Canny edge detection & region segmentation\n"
            "  features  : SIFT keypoint detection and descriptor extraction\n"
            "  match     : Pairwise SIFT feature matching with Lowe's ratio test\n"
            "  geometry  : Homography matrix estimation (RANSAC) and perspective warping\n"
            "  epipolar  : Fundamental matrix estimation (RANSAC) and epipolar lines\n"
            "  depth     : Stereo disparity (StereoSGBM) and metric depth recovery\n"
            "  full      : Execute end-to-end analysis pipeline"
        )
    )

    # Input specifications
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to single input image (for modes: enhance, edges, features, full)."
    )
    parser.add_argument(
        "--left",
        type=str,
        default=None,
        help="Path to left view of stereo/image pair (for modes: match, geometry, epipolar, depth, full)."
    )
    parser.add_argument(
        "--right",
        type=str,
        default=None,
        help="Path to right view of stereo/image pair (for modes: match, geometry, epipolar, depth, full)."
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to store generated image artifacts and JSON analysis report (default: 'outputs')."
    )

    # Stereo Camera Parameters
    parser.add_argument(
        "--focal-length",
        type=float,
        default=800.0,
        help="Camera focal length in pixels (default: 800.0)."
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.1,
        help="Stereo camera baseline distance in meters (default: 0.1)."
    )
    parser.add_argument(
        "--stereo-algo",
        type=str,
        choices=["sgbm", "bm"],
        default="sgbm",
        help="Stereo correspondence algorithm: 'sgbm' (Semi-Global) or 'bm' (Block Matching) (default: sgbm)."
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=64,
        help="Maximum disparity search range (must be multiple of 16, default: 64)."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=7,
        help="Matching block size (must be an odd integer >= 3, default: 7)."
    )

    # Matching options
    parser.add_argument(
        "--matcher",
        type=str,
        choices=["flann", "bf"],
        default="flann",
        help="Descriptor matching algorithm: 'flann' or 'bf' (default: flann)."
    )
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold (default: 0.75)."
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed diagnostic console logging."
    )

    return parser


def run_enhance_mode(img_path: str, out_dir: Path, report: AnalysisReport, verbose: bool = False) -> None:
    """Execute Module 1: Preprocessing and Enhancement."""
    print(f"\n[CV-Scope] Running Preprocessing & Enhancement on '{img_path}'...")
    img = load_image(img_path)

    report.add_algorithm("Grayscale Conversion (Rec.601)")
    report.add_algorithm("Gaussian Smoothing Filter")
    report.add_algorithm("Median Denoising Filter")
    report.add_algorithm("Histogram Equalization")
    report.add_algorithm("CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    report.add_algorithm("Unsharp Masking Spatial Convolution")

    prep_res = run_preprocessing(img)
    enh_res = run_enhancement(prep_res["grayscale"])

    report.record_time("Preprocessing", prep_res["elapsed_ms"])
    report.record_time("Enhancement", enh_res["elapsed_ms"])
    report.update_metrics("Image_Dimensions", {"width": img.shape[1], "height": img.shape[0], "channels": img.shape[2] if len(img.shape) == 3 else 1})
    report.update_metrics("Preprocessing_Stats", prep_res["statistics"])
    report.update_metrics("Enhancement_Stats", enh_res["metrics"])

    # Save artifact images
    save_image(out_dir / "grayscale.png", prep_res["grayscale"])
    save_image(out_dir / "gaussian_denoised.png", prep_res["gaussian_denoised"])
    save_image(out_dir / "median_denoised.png", prep_res["median_denoised"])
    save_image(out_dir / "clahe.png", enh_res["clahe"])
    save_image(out_dir / "sharpened.png", enh_res["sharpened"])

    # Save comparison figure
    plot_enhancement_comparison(
        prep_res["original"],
        prep_res["grayscale"],
        prep_res["gaussian_denoised"],
        enh_res["clahe"],
        enh_res["sharpened"],
        out_dir / "enhanced.png"
    )

    print(f"  -> Preprocessing Time : {prep_res['elapsed_ms']} ms")
    print(f"  -> Enhancement Time   : {enh_res['elapsed_ms']} ms")
    print(f"  -> Contrast Ratio     : {prep_res['statistics']['contrast_ratio']} (raw) -> {enh_res['metrics']['clahe']['contrast_ratio']} (CLAHE)")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_edges_mode(img_path: str, out_dir: Path, report: AnalysisReport, verbose: bool = False) -> None:
    """Execute Module 2: Edge Detection and Segmentation Analysis."""
    print(f"\n[CV-Scope] Running Edge Detection & Segmentation on '{img_path}'...")
    img = load_image(img_path)
    gray = to_grayscale(img)

    report.add_algorithm("Sobel Directional Spatial Derivatives")
    report.add_algorithm("Gradient Magnitude Field")
    report.add_algorithm("Canny Edge Detector (Hysteresis & Non-Maximum Suppression)")
    report.add_algorithm("Otsu Optimal Bimodal Thresholding")
    report.add_algorithm("Morphological Opening & Closing")
    report.add_algorithm("Connected Component Region Labeling")

    seg_res = run_edge_segmentation(gray)

    report.record_time("Edge_and_Segmentation", seg_res["elapsed_ms"])
    report.update_metrics("Edge_Density", {"ratio": seg_res["edge_density"], "percentage": round(seg_res["edge_density"] * 100, 2)})
    report.update_metrics("Segmentation", {
        "num_connected_components": seg_res["num_regions"],
        "optimal_threshold_otsu": seg_res["optimal_threshold"]
    })

    # Save artifacts
    save_image(out_dir / "sobel_x.png", seg_res["sobel"]["sobel_x_uint8"])
    save_image(out_dir / "sobel_y.png", seg_res["sobel"]["sobel_y_uint8"])
    save_image(out_dir / "sobel.png", seg_res["sobel"]["magnitude_uint8"])
    save_image(out_dir / "canny.png", seg_res["canny"])
    save_image(out_dir / "segmentation.png", seg_res["colored_regions"])

    plot_edge_segmentation_grid(
        seg_res["sobel"]["sobel_x_uint8"],
        seg_res["sobel"]["sobel_y_uint8"],
        seg_res["sobel"]["magnitude_uint8"],
        seg_res["canny"],
        seg_res["binary_threshold"],
        seg_res["colored_regions"],
        out_dir / "edge_segmentation_summary.png"
    )

    print(f"  -> Execution Time     : {seg_res['elapsed_ms']} ms")
    print(f"  -> Edge Pixel Density : {round(seg_res['edge_density'] * 100, 2)}%")
    print(f"  -> Otsu Threshold     : {seg_res['optimal_threshold']}")
    print(f"  -> Detected Regions   : {seg_res['num_regions']} components")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_features_mode(img_path: str, out_dir: Path, report: AnalysisReport, verbose: bool = False) -> None:
    """Execute Module 3 (Single-Image): SIFT Feature Extraction."""
    print(f"\n[CV-Scope] Running SIFT Feature Extraction on '{img_path}'...")
    img = load_image(img_path)

    report.add_algorithm("SIFT Keypoint Detection (DoG Extrema)")
    report.add_algorithm("SIFT 128-D Gradient Orientation Descriptor Extraction")

    feat_res = run_feature_analysis(img)

    report.record_time("Feature_Extraction", feat_res["elapsed_ms"])
    report.update_metrics("Features", {"num_keypoints": feat_res["num_kps1"]})

    save_image(out_dir / "sift_keypoints.png", feat_res["kps1_visual"])

    print(f"  -> Execution Time     : {feat_res['elapsed_ms']} ms")
    print(f"  -> SIFT Keypoints     : {feat_res['num_kps1']}")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_match_mode(left_path: str, right_path: str, out_dir: Path, report: AnalysisReport, config: PipelineConfig, verbose: bool = False) -> Dict[str, Any]:
    """Execute Module 3 (Pairwise): SIFT Matching with Lowe's Ratio Test."""
    print(f"\n[CV-Scope] Running SIFT Feature Matching between '{left_path}' and '{right_path}'...")
    img1 = load_image(left_path)
    img2 = load_image(right_path)

    report.add_algorithm("SIFT Feature Detection (Pairwise)")
    report.add_algorithm(f"Feature Matching ({config.features.matcher_type.upper()})")
    report.add_algorithm(f"Lowe's Ratio Test (tau={config.features.ratio_threshold})")

    feat_res = run_feature_analysis(img1, img2, config=config.features)

    report.record_time("Feature_Matching", feat_res["elapsed_ms"])
    report.update_metrics("Feature_Matching", {
        "keypoints_image1": feat_res["num_kps1"],
        "keypoints_image2": feat_res["num_kps2"],
        "raw_knn_matches": feat_res["raw_matches_count"],
        "good_matches": feat_res["good_matches_count"],
        "match_ratio": feat_res["match_ratio"]
    })

    save_image(out_dir / "feature_matches.png", feat_res["matches_visual"])

    print(f"  -> Execution Time     : {feat_res['elapsed_ms']} ms")
    print(f"  -> Left Keypoints     : {feat_res['num_kps1']}")
    print(f"  -> Right Keypoints    : {feat_res['num_kps2']}")
    print(f"  -> Good Matches       : {feat_res['good_matches_count']} (Ratio: {feat_res['match_ratio']})")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")

    return feat_res


def run_geometry_mode(left_path: str, right_path: str, out_dir: Path, report: AnalysisReport, config: PipelineConfig, feat_res: Optional[Dict[str, Any]] = None, verbose: bool = False) -> None:
    """Execute Module 4: Planar Homography & Perspective Warping."""
    print(f"\n[CV-Scope] Running Geometric Analysis & Homography Estimation...")
    img1 = load_image(left_path)
    img2 = load_image(right_path)

    if feat_res is None or "pts1" not in feat_res:
        feat_res = run_feature_analysis(img1, img2, config=config.features)

    pts1, pts2 = feat_res["pts1"], feat_res["pts2"]
    if len(pts1) < 4:
        raise ValueError(f"Insufficient corresponding points for Homography (need >= 4, found {len(pts1)}).")

    report.add_algorithm("Homography Matrix Estimation (Direct Linear Transform)")
    report.add_algorithm(f"RANSAC Outlier Rejection (thresh={config.geometry.ransac_reproj_threshold}px)")
    report.add_algorithm("Perspective Image Warping & Alpha-Blended Registration")

    geom_res = run_geometry_analysis(img1, img2, pts1, pts2, config=config.geometry)

    report.record_time("Geometry_and_Homography", geom_res["elapsed_ms"])
    report.update_metrics("Geometry", {
        "inliers": geom_res["num_inliers"],
        "inlier_ratio": geom_res["inlier_ratio"],
        "mean_reprojection_error_px": geom_res["reprojection_error"],
        "homography_matrix": geom_res["homography_matrix"]
    })

    save_image(out_dir / "warped_image.png", geom_res["warped_image"])
    save_image(out_dir / "alignment_overlay.png", geom_res["alignment_overlay"])

    plot_geometry_alignment(
        img2,
        geom_res["warped_image"],
        geom_res["alignment_overlay"],
        out_dir / "homography_warp.png"
    )

    print(f"  -> Execution Time     : {geom_res['elapsed_ms']} ms")
    print(f"  -> RANSAC Inliers     : {geom_res['num_inliers']} / {len(pts1)} (Ratio: {geom_res['inlier_ratio']})")
    print(f"  -> Mean Reproj Error  : {geom_res['reprojection_error']} px")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_epipolar_mode(left_path: str, right_path: str, out_dir: Path, report: AnalysisReport, config: PipelineConfig, feat_res: Optional[Dict[str, Any]] = None, verbose: bool = False) -> None:
    """Execute Module 5: Fundamental Matrix & Epipolar Geometry."""
    print(f"\n[CV-Scope] Running Epipolar Geometry & Fundamental Matrix Estimation...")
    img1 = load_image(left_path)
    img2 = load_image(right_path)

    if feat_res is None or "pts1" not in feat_res:
        feat_res = run_feature_analysis(img1, img2, config=config.features)

    pts1, pts2 = feat_res["pts1"], feat_res["pts2"]
    if len(pts1) < 8:
        raise ValueError(f"Insufficient corresponding points for Fundamental Matrix (need >= 8, found {len(pts1)}).")

    report.add_algorithm("Fundamental Matrix Estimation (8-Point Algorithm)")
    report.add_algorithm(f"RANSAC Epipolar Line Distance Rejection (thresh={config.epipolar.ransac_threshold}px)")
    report.add_algorithm("Epipolar Line Computation & Conjugate Mapping")

    epi_res = run_epipolar_analysis(img1, img2, pts1, pts2, config=config.epipolar)

    report.record_time("Epipolar_Geometry", epi_res["elapsed_ms"])
    report.update_metrics("Epipolar", {
        "inliers": epi_res["num_inliers"],
        "inlier_ratio": epi_res["inlier_ratio"],
        "fundamental_matrix": epi_res["fundamental_matrix"]
    })

    save_image(out_dir / "epipolar_lines.png", epi_res["epipolar_visual"])
    plot_epipolar_lines(epi_res["epipolar_visual"], out_dir / "epipolar_composite.png")

    print(f"  -> Execution Time     : {epi_res['elapsed_ms']} ms")
    print(f"  -> Epipolar Inliers   : {epi_res['num_inliers']} / {len(pts1)} (Ratio: {epi_res['inlier_ratio']})")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_depth_mode(left_path: str, right_path: str, out_dir: Path, report: AnalysisReport, config: PipelineConfig, verbose: bool = False) -> None:
    """Execute Module 6: Stereo Disparity & Depth Estimation."""
    print(f"\n[CV-Scope] Running Stereo Correspondence & Depth Reconstruction...")
    img_l = load_image(left_path)
    img_r = load_image(right_path)

    report.add_algorithm(f"Stereo Disparity Estimation ({config.stereo.algorithm.upper()})")
    report.add_algorithm("Disparity Normalization & False-Color Mapping (Inferno)")
    report.add_algorithm("Triangulation Depth Recovery (Z = f * B / d)")

    stereo_res = run_stereo_depth_pipeline(img_l, img_r, config=config.stereo)

    report.record_time("Stereo_Depth", stereo_res["elapsed_ms"])
    report.update_metrics("Disparity_Statistics", stereo_res["disparity_stats"])
    report.update_metrics("Depth_Statistics", stereo_res["depth_stats"])

    # Save visual outputs
    save_image(out_dir / "disparity_raw_normalized.png", stereo_res["disparity_normalized"])
    save_image(out_dir / "disparity.png", stereo_res["disparity_colored"])
    save_image(out_dir / "depth.png", stereo_res["depth_colored"])

    plot_disparity_and_depth(
        img_l,
        stereo_res["disparity_colored"],
        stereo_res["depth_colored"],
        stereo_res["disparity"],
        stereo_res["depth"],
        out_dir / "disparity_depth_analysis.png"
    )

    d_stats = stereo_res["disparity_stats"]
    z_stats = stereo_res["depth_stats"]
    print(f"  -> Execution Time     : {stereo_res['elapsed_ms']} ms")
    print(f"  -> Disparity Range    : [{d_stats['min_disparity']}px - {d_stats['max_disparity']}px] (Mean: {d_stats['mean_disparity']}px)")
    print(f"  -> Valid Disparity    : {d_stats['valid_percentage']}% of pixels")
    print(f"  -> Depth Recovery (Z) : [{z_stats['min_depth']} - {z_stats['max_depth']}] (Mean: {z_stats['mean_depth']}, Type: {z_stats['depth_type']})")
    print(f"  -> Artifacts Saved in : {out_dir.resolve()}")


def run_full_pipeline(
    single_path: Optional[str],
    left_path: Optional[str],
    right_path: Optional[str],
    out_dir: Path,
    report: AnalysisReport,
    config: PipelineConfig,
    verbose: bool = False
) -> None:
    """Execute the complete CV-Scope pipeline sequentially."""
    print("=" * 60)
    print("  CV-Scope: EXECUTING FULL COMPUTER VISION PIPELINE")
    print("=" * 60)

    # Determine primary input for single-image operations
    primary_img_path = single_path if single_path else left_path
    if primary_img_path is None:
        raise ValueError("Full pipeline requires at least --input or --left/--right stereo pair.")

    # 1. Preprocessing & Enhancement
    run_enhance_mode(primary_img_path, out_dir, report, verbose=verbose)

    # 2. Edge & Segmentation
    run_edges_mode(primary_img_path, out_dir, report, verbose=verbose)

    # 3. Single-Image Features
    run_features_mode(primary_img_path, out_dir, report, verbose=verbose)

    # If stereo pair is provided, execute two-view geometry & depth modules
    if left_path and right_path:
        print("\n--- Stereo Image Pair Provided: Executing Two-View Geometry Modules ---")
        # 4. Pairwise Feature Matching
        feat_res = run_match_mode(left_path, right_path, out_dir, report, config, verbose=verbose)

        # 5. Homography & Geometric Alignment
        try:
            run_geometry_mode(left_path, right_path, out_dir, report, config, feat_res=feat_res, verbose=verbose)
        except Exception as e:
            report.add_warning(f"Geometry analysis skipped: {str(e)}")
            print(f"  [!] Warning: Geometry analysis skipped: {e}")

        # 6. Epipolar Geometry & Fundamental Matrix
        try:
            run_epipolar_mode(left_path, right_path, out_dir, report, config, feat_res=feat_res, verbose=verbose)
        except Exception as e:
            report.add_warning(f"Epipolar analysis skipped: {str(e)}")
            print(f"  [!] Warning: Epipolar analysis skipped: {e}")

        # 7. Stereo Disparity & Depth Estimation
        try:
            run_depth_mode(left_path, right_path, out_dir, report, config, verbose=verbose)
        except Exception as e:
            report.add_warning(f"Stereo depth pipeline failed: {str(e)}")
            print(f"  [!] Warning: Stereo depth pipeline failed: {e}")
    else:
        print("\n--- Note: Single image provided. Skipping stereo-specific modules (matching, homography, epipolar, depth). ---")
        report.add_warning("Stereo-specific modules skipped because only a single image was provided.")


def main() -> int:
    """CLI application entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    # Initialize configuration
    config = PipelineConfig(
        output_dir=str(out_dir),
        features=FeatureConfig(
            matcher_type=args.matcher,
            ratio_threshold=args.ratio_thresh
        ),
        stereo=StereoConfig(
            algorithm=args.stereo_algo,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            focal_length=args.focal_length,
            baseline=args.baseline
        )
    )

    inputs = {
        "input": args.input,
        "left": args.left,
        "right": args.right
    }
    report = AnalysisReport(mode=args.mode, input_files=inputs)

    try:
        if args.mode == "enhance":
            if not args.input:
                raise ValueError("Mode 'enhance' requires '--input <path>'.")
            run_enhance_mode(args.input, out_dir, report, verbose=args.verbose)

        elif args.mode == "edges":
            if not args.input:
                raise ValueError("Mode 'edges' requires '--input <path>'.")
            run_edges_mode(args.input, out_dir, report, verbose=args.verbose)

        elif args.mode == "features":
            if not args.input:
                raise ValueError("Mode 'features' requires '--input <path>'.")
            run_features_mode(args.input, out_dir, report, verbose=args.verbose)

        elif args.mode == "match":
            if not args.left or not args.right:
                raise ValueError("Mode 'match' requires both '--left <path>' and '--right <path>'.")
            run_match_mode(args.left, args.right, out_dir, report, config, verbose=args.verbose)

        elif args.mode == "geometry":
            if not args.left or not args.right:
                raise ValueError("Mode 'geometry' requires both '--left <path>' and '--right <path>'.")
            run_geometry_mode(args.left, args.right, out_dir, report, config, verbose=args.verbose)

        elif args.mode == "epipolar":
            if not args.left or not args.right:
                raise ValueError("Mode 'epipolar' requires both '--left <path>' and '--right <path>'.")
            run_epipolar_mode(args.left, args.right, out_dir, report, config, verbose=args.verbose)

        elif args.mode == "depth":
            if not args.left or not args.right:
                raise ValueError("Mode 'depth' requires both '--left <path>' and '--right <path>'.")
            run_depth_mode(args.left, args.right, out_dir, report, config, verbose=args.verbose)

        elif args.mode == "full":
            run_full_pipeline(
                single_path=args.input,
                left_path=args.left,
                right_path=args.right,
                out_dir=out_dir,
                report=report,
                config=config,
                verbose=args.verbose
            )

        report.finalize(success=True)
        report_file = report.save_json(out_dir / "analysis_report.json")
        print("\n" + "=" * 60)
        print(f"SUCCESS: Analysis complete. JSON report saved to:")
        print(f"         {report_file}")
        print("=" * 60 + "\n")
        return 0

    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR: File Not Found] {fnf_err}", file=sys.stderr)
        report.finalize(success=False)
        report.add_warning(str(fnf_err))
        report.save_json(out_dir / "analysis_report.json")
        return 1

    except ValueError as val_err:
        print(f"\n[ERROR: Invalid Input / Parameter] {val_err}", file=sys.stderr)
        report.finalize(success=False)
        report.add_warning(str(val_err))
        report.save_json(out_dir / "analysis_report.json")
        return 1

    except Exception as exc:
        print(f"\n[ERROR: Execution Failure] {exc}", file=sys.stderr)
        report.finalize(success=False)
        report.add_warning(str(exc))
        report.save_json(out_dir / "analysis_report.json")
        return 2


if __name__ == "__main__":
    sys.exit(main())
