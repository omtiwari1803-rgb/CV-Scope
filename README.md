# CV-Scope: A Computer Vision-Based Scene Geometry and Depth Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)

An academic-grade, modular, command-line Computer Vision system demonstrating classical computer vision principles from first principles—free from opaque deep learning models.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Key Features](#4-key-features)
5. [Computer Vision Concepts Used](#5-computer-vision-concepts-used)
6. [System Architecture](#6-system-architecture)
7. [Project Structure](#7-project-structure)
8. [Requirements & Installation](#8-requirements--installation)
9. [Dataset Setup](#9-dataset-setup)
10. [Command-Line Interface (CLI) Usage](#10-command-line-interface-cli-usage)
11. [Execution Examples](#11-execution-examples)
12. [Output Artifacts Description](#12-output-artifacts-description)
13. [Testing Instructions](#13-testing-instructions)
14. [Detailed Algorithm Explanations](#14-detailed-algorithm-explanations)
15. [Limitations](#15-limitations)
16. [Future Enhancements](#16-future-enhancements)

---

## 1. Project Overview

**CV-Scope** is a classical Computer Vision analysis framework developed for academic course evaluation. It provides a complete, deterministic, and interpretable suite of classical computer vision algorithms operating on single images and binocular stereo pairs.

The system is designed for **headless command-line execution** without requiring GUIs, notebook servers, or IDE configurations. Visualizations and machine-readable JSON reports are automatically saved to disk.

---

## 2. Problem Statement

While deep neural networks have achieved high benchmarks on computer vision tasks, they often obscure the foundational physical, geometric, and statistical mechanics that govern digital imagery. In safety-critical fields such as autonomous robotics, aerial photogrammetry, and industrial metrology, engineers must understand the exact mathematical relationship between light projection, feature geometry, and 3D scene reconstruction.

**CV-Scope** demonstrates that robust scene analysis, geometric registration, and 3D depth recovery can be achieved deterministically through classical computer vision principles without requiring black-box neural networks.

---

## 3. Objectives

- **Educational Transparency**: Provide clean, modular Python implementations of classical vision algorithms with comprehensive comments.
- **Strictly Classical Operations**: Rely exclusively on deterministic signal processing and geometric algorithms in OpenCV, NumPy, and scikit-image.
- **Robust Error Handling**: Handle degenerate configurations, missing descriptors, collinear points, and invalid disparities gracefully with clear error messages.
- **Empirical Rigor**: Provide automated unit tests (`pytest`) and verifiable telemetry in machine-readable JSON format.

---

## 4. Key Features

- **Multi-Mode CLI**: Execute isolated algorithmic components or run an end-to-end full pipeline.
- **Headless Visualization Engine**: Uses Matplotlib's `Agg` backend to generate publication-quality figures directly to disk without GUI dependencies.
- **Metric and Relative Depth Analysis**: Implements the triangulation formula $Z = \frac{f \cdot B}{d}$, explicitly differentiating between calibrated metric depth (meters) and relative disparity scaling.
- **Automated Telemetry**: Accumulates processing latency, keypoint statistics, inlier ratios, and depth distributions into `outputs/analysis_report.json`.

---

## 5. Computer Vision Concepts Used

| Concept Area | Algorithms & Techniques Implemented |
| :--- | :--- |
| **Preprocessing** | Grayscale conversion (Rec. 601), Gaussian smoothing, Median filtering, Min-Max normalization |
| **Enhancement** | Global Histogram Equalization, CLAHE (Contrast Limited Adaptive Histogram Equalization), Unsharp Masking |
| **Edge Detection** | Sobel horizontal/vertical derivatives, Gradient magnitude & orientation, Canny 4-stage edge detection |
| **Segmentation** | Otsu optimal thresholding, Morphological opening/closing, Connected-component spatial labeling |
| **Features** | SIFT (Scale-Invariant Feature Transform) keypoint detection and 128-D descriptor extraction |
| **Correspondence** | FLANN / Brute-Force matching, k-Nearest Neighbors ($k=2$), Lowe's ambiguity ratio test |
| **Image Geometry** | Planar homography estimation, RANSAC outlier elimination, Perspective spatial warping |
| **Epipolar Geometry** | Fundamental matrix ($F$), Epipolar constraint ($x_2^T F x_1 = 0$), Conjugate epipolar lines |
| **Stereo Depth** | Semi-Global Block Matching (StereoSGBM), Disparity normalization, Triangulation depth ($Z = \frac{f \cdot B}{d}$) |

---

## 6. System Architecture

CV-Scope is organized as a decoupled, layered pipeline where each module operates independently and communicates via standard NumPy arrays and typed dataclasses.

```
                  +-----------------------------------+
                  |        main.py (CLI Controller)   |
                  +-----------------+-----------------+
                                    |
     +------------------------------+------------------------------+
     |                              |                              |
     v                              v                              v
[Low-Level Vision]         [Mid-Level Vision]            [Multi-View Geometry]
- preprocessing.py         - segmentation.py             - features.py
- enhancement.py                                         - geometry.py
                                                         - epipolar.py
                                                         - stereo.py
     |                              |                              |
     +------------------------------+------------------------------+
                                    |
                    +---------------+---------------+
                    v                               v
          [visualization.py]                  [reporting.py]
          - Matplotlib Agg figures            - analysis_report.json
```

For complete UML diagrams (Class, Component, Sequence, and Use Case diagrams), see [docs/architecture.md](file:///docs/architecture.md).

---

## 7. Project Structure

```
CV-Scope/
├── README.md                      # Primary project guide and documentation
├── statement.md                   # Formal academic problem statement
├── requirements.txt               # Dependencies (NumPy, OpenCV, Matplotlib, scikit-image, pytest)
├── config.py                      # Central algorithm parameters & dataclasses
├── main.py                        # CLI controller implementing all 8 modes
│
├── src/                           # Modular computer vision implementations
│   ├── __init__.py
│   ├── utils.py                   # Image I/O, validation, Timer context manager
│   ├── preprocessing.py           # Grayscale, resize, Gaussian/median filtering, normalization
│   ├── enhancement.py             # Histogram equalization, CLAHE, unsharp masking
│   ├── segmentation.py            # Sobel, Canny, Otsu thresholding, morphology, connected components
│   ├── features.py                # SIFT detector, FLANN/BF matching, Lowe's ratio test
│   ├── geometry.py                # Homography estimation (RANSAC), perspective warp, alignment
│   ├── epipolar.py                # Fundamental matrix (RANSAC), epipolar lines computation & drawing
│   ├── stereo.py                  # StereoSGBM/StereoBM disparity, normalization, depth Z = f*B/d
│   ├── visualization.py           # Headless Matplotlib figures, composite panels, colorbars
│   └── reporting.py               # JSON metrics accumulator and file serializer
│
├── tests/                         # Automated unit & integration tests
│   ├── __init__.py
│   ├── test_preprocessing.py      # Tests for low-level filtering & enhancement
│   ├── test_features.py           # Tests for SIFT extraction, descriptors, matching
│   ├── test_geometry.py           # Tests for homography, RANSAC, warping
│   ├── test_stereo.py             # Tests for epipolar geometry, disparity, depth formula
│   └── test_cli.py                # Subprocess integration tests for CLI modes and error handling
│
├── data/                          # Royalty-free benchmark datasets
│   ├── single/                    # Geometric multi-textured test image
│   │   └── sample.png
│   └── stereo/                    # Calibrated stereo pair with exact horizontal parallax
│       ├── left.png
│       └── right.png
│
├── outputs/                       # Destination folder for generated figures and JSON reports
│   └── .gitkeep
│
├── docs/                          # Detailed academic documentation
│   ├── architecture.md            # UML diagrams, component design, NFRs
│   └── workflow.md                # Mathematical derivations & theoretical models
│
└── scripts/                       # Dataset synthesis scripts
    └── generate_sample_data.py
```

---

## 8. Requirements & Installation

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **pip** package installer

### Installation Steps

1. Clone or navigate to the repository directory:
   ```bash
   cd "CV-Scope"
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 9. Dataset Setup

CV-Scope includes royalty-free, geometrically exact test datasets generated deterministically:
- **`data/single/sample.png`**: Multi-contrast image featuring concentric rings, checkerboards, gradients, and polygon edges.
- **`data/stereo/left.png` and `data/stereo/right.png`**: Calibrated stereo pair projecting a 3D scene across three distinct depth planes ($Z_{\text{fg}} = 0.8\text{m}$, $Z_{\text{mid}} = 1.33\text{m}$, $Z_{\text{bg}} = 2.5\text{m}$) yielding precise horizontal disparities.

To regenerate or verify the dataset:
```bash
python scripts/generate_sample_data.py
```

---

## 10. Command-Line Interface (CLI) Usage

### Display Help and Options
```bash
python main.py --help
```

### Supported Modes

| Mode Flag | Purpose | Required Arguments |
| :--- | :--- | :--- |
| `--mode enhance` | Preprocessing, denoising, histogram equalization, CLAHE | `--input <path>` |
| `--mode edges` | Sobel X/Y, gradient magnitude, Canny, Otsu segmentation | `--input <path>` |
| `--mode features` | SIFT keypoint detection and descriptor extraction | `--input <path>` |
| `--mode match` | Pairwise SIFT feature matching with Lowe's ratio test | `--left <path> --right <path>` |
| `--mode geometry` | Homography estimation via RANSAC and perspective warp | `--left <path> --right <path>` |
| `--mode epipolar` | Fundamental matrix ($F$) and conjugate epipolar lines | `--left <path> --right <path>` |
| `--mode depth` | Stereo disparity (StereoSGBM) and metric depth recovery | `--left <path> --right <path>` |
| `--mode full` | Sequential execution of all applicable modules | `--left <path> --right <path>` or `--input <path>` |

### Configurable Options
- `--output <dir>`: Target directory for artifacts (default: `outputs/`).
- `--focal-length <float>`: Camera focal length in pixels (default: `800.0`).
- `--baseline <float>`: Stereo baseline distance in meters (default: `0.1`).
- `--stereo-algo {sgbm,bm}`: Disparity correspondence algorithm (default: `sgbm`).
- `--num-disparities <int>`: Disparity search range (multiple of 16, default: `64`).
- `--block-size <int>`: Matching window size (odd integer $\ge 3$, default: `7`).
- `--matcher {flann,bf}`: Descriptor matching algorithm (default: `flann`).
- `--ratio-thresh <float>`: Lowe's ratio threshold (default: `0.75`).
- `--verbose`: Enable detailed console diagnostic output.

---

## 11. Execution Examples

### Example 1: Image Enhancement
```bash
python main.py --mode enhance --input data/single/sample.png
```
*Outputs: `grayscale.png`, `gaussian_denoised.png`, `median_denoised.png`, `clahe.png`, `sharpened.png`, `enhanced.png`.*

### Example 2: Edge Detection & Segmentation
```bash
python main.py --mode edges --input data/single/sample.png
```
*Outputs: `sobel_x.png`, `sobel_y.png`, `sobel.png`, `canny.png`, `segmentation.png`, `edge_segmentation_summary.png`.*

### Example 3: SIFT Keypoint Extraction
```bash
python main.py --mode features --input data/single/sample.png
```
*Outputs: `sift_keypoints.png`.*

### Example 4: Pairwise Feature Matching
```bash
python main.py --mode match --left data/stereo/left.png --right data/stereo/right.png
```
*Outputs: `feature_matches.png`.*

### Example 5: Planar Homography & Projective Alignment
```bash
python main.py --mode geometry --left data/stereo/left.png --right data/stereo/right.png
```
*Outputs: `warped_image.png`, `alignment_overlay.png`, `homography_warp.png`.*

### Example 6: Epipolar Geometry & Fundamental Matrix
```bash
python main.py --mode epipolar --left data/stereo/left.png --right data/stereo/right.png
```
*Outputs: `epipolar_lines.png`, `epipolar_composite.png`.*

### Example 7: Stereo Disparity & Depth Estimation
```bash
python main.py --mode depth --left data/stereo/left.png --right data/stereo/right.png --focal-length 800.0 --baseline 0.1
```
*Outputs: `disparity.png`, `depth.png`, `disparity_depth_analysis.png`.*

### Example 8: Full Pipeline Execution
```bash
python main.py --mode full --left data/stereo/left.png --right data/stereo/right.png --output outputs/
```
*Executes all modules in logical sequence and exports `outputs/analysis_report.json`.*

---

## 12. Output Artifacts Description

When executed, the system populates the output folder with the following files:

| File Name | Description |
| :--- | :--- |
| `grayscale.png` | Standardized single-channel 8-bit luminance image. |
| `enhanced.png` | 2x3 panel comparing Original, Grayscale, Gaussian, CLAHE, Sharpening, and Histograms. |
| `sobel.png` | Normalized spatial gradient magnitude field $\sqrt{G_x^2 + G_y^2}$. |
| `canny.png` | Binary edge map produced via non-maximum suppression and hysteresis. |
| `segmentation.png` | Pseudo-colored connected components labeled by spatial connectivity. |
| `sift_keypoints.png` | Rich SIFT keypoint visualization with scale circles and gradient orientation vectors. |
| `feature_matches.png` | Side-by-side correspondence lines filtered via Lowe's ratio test. |
| `homography_warp.png` | 3-panel display: Reference image, Warped source image ($H \cdot I_1$), and Alpha blend. |
| `epipolar_lines.png` | Epipolar lines and matching feature points across left and right views. |
| `disparity.png` | High-contrast false-color disparity map (Inferno colormap). |
| `depth.png` | Metric depth map (Viridis colormap) computed via $Z = \frac{f \cdot B}{d}$. |
| `analysis_report.json` | Comprehensive telemetry report with numerical metrics, execution times, and timestamps. |

---

## 13. Testing Instructions

The test suite uses `pytest` and verifies unit functions, edge cases, mathematical formulas, and CLI subprocess execution.

Run the test suite:
```bash
python -m pytest -v
```

Expected output:
```
tests/test_cli.py::test_cli_help PASSED
tests/test_cli.py::test_cli_enhance_mode PASSED
tests/test_cli.py::test_cli_edges_mode PASSED
...
tests/test_stereo.py::test_compute_depth_formula PASSED
tests/test_stereo.py::test_normalize_disparity_for_visualization PASSED

============================= 40 passed in 14.21s =============================
```

---

## 14. Detailed Algorithm Explanations

### Why SIFT?
Scale-Invariant Feature Transform (Lowe, 2004) detects extrema in Difference-of-Gaussians (DoG) scale space and characterizes keypoint neighborhoods using 128-D orientation histograms. SIFT was selected because it is invariant to image scale, rotation, and affine illumination variations, outperforming corner detectors (e.g., Harris) on real-world stereo scenes.

### Why RANSAC?
Feature matching inherently produces false matches (outliers) from repetitive patterns or occlusions. Ordinary least-squares regression degrades catastrophically in the presence of even a single outlier. RANSAC (Random Sample Consensus) iteratively samples minimal sets (4 points for homography, 8 points for fundamental matrix) to discover the true geometric consensus set, tolerating up to 50%+ outlier contamination.

### Why Canny?
Unlike simple gradient thresholding which produces thick, discontinuous edge blobs, Canny edge detection applies directional non-maximum suppression (NMS) to thin boundaries to single-pixel ridges, followed by hysteresis thresholding with dual thresholds to trace faint continuous contours.

### Why StereoSGBM?
Classical Block Matching (StereoBM) relies strictly on local rectangular windows, which struggle with textureless regions and create boundary bleeding artifacts. StereoSGBM (Semi-Global Block Matching) approximates 2D MRF energy minimization along 1D paths from multiple directions, effectively penalizing disparity changes ($P_1, P_2$) while preserving sharp depth discontinuities.

### Why Homography?
Homography models 2D planar projective transformations ($x_2 \sim H x_1$), suitable for aligning planar surfaces or rotating camera views. In non-planar 3D scenes, Homography is complemented by the Fundamental Matrix ($x_2^T F x_1 = 0$) to model general epipolar geometry.

---

## 15. Limitations

1. **Untextured Surfaces**: Block-based stereo algorithms rely on intensity variance; completely homogeneous walls produce disparity voids.
2. **Extreme Baseline & Occlusions**: Large baseline separations introduce wide non-overlapping occlusion zones where no stereo correspondence exists.
3. **Planar Constraint for Homography**: A single $3 \times 3$ homography matrix cannot model non-planar parallax in complex 3D scenes; epipolar geometry is required for general multi-view structures.

---

## 16. Future Enhancements

- **Sub-pixel Refinement**: Parabolic sub-pixel interpolation on stereo cost volumes.
- **Bundle Adjustment**: Non-linear Levenberg-Marquardt optimization over multi-view camera poses and 3D point landmarks.
- **Dense Point Cloud Export**: Exporting triangulated $(X, Y, Z)$ coordinates into standard `.ply` or `.pcd` point cloud formats for 3D visualization in MeshLab.
- **Multi-Baseline Fusion**: Integrating three or more camera views to eliminate stereo occlusion ambiguity.
