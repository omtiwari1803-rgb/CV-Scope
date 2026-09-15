# Project Statement: CV-Scope

**CV-Scope: A Computer Vision-Based Scene Geometry and Depth Analysis System**  
*Academic Course Evaluation Specification Document*

---

## 1. Problem Statement

Modern deep learning methods have achieved remarkable success in high-level vision tasks; however, deep neural networks function largely as black-box approximators, frequently obscuring the foundational physical, geometric, and statistical principles of computer vision. In robotics, autonomous navigation, industrial visual inspection, and photogrammetry, critical tasks require deterministic geometric reasoning, strict error bounds, and interpretable physical relationships. 

Moreover, academic evaluation in computer vision demands an explicit grasp of classical algorithms: how high-frequency edge gradients are formed, how projective transformations govern multi-view correspondences, how epipolar geometry constrains 3D search spaces, and how binocular disparity directly yields depth through triangulation.

**CV-Scope** addresses this educational and operational need by providing a fully modular, transparent, and reproducible command-line computer vision system. It implements classical algorithms from first principles using standard mathematical libraries (OpenCV, NumPy, scikit-image) to perform complete scene geometry and metric depth analysis without relying on opaque deep neural networks.

---

## 2. Project Scope

CV-Scope encompasses:
1. **Low-Level Vision**: Intensity standardization, non-linear filtering, contrast optimization via global histogram equalization, and adaptive local contrast enhancement (CLAHE).
2. **Mid-Level Vision**: Directional differential calculus (Sobel operators), multi-stage hysteresis boundary extraction (Canny), statistical bi-level segmentation (Otsu), and mathematical morphology for structural refinement.
3. **Feature Extraction & Description**: Difference-of-Gaussians scale-space extrema detection and orientation-invariant 128-dimensional gradient histogram description (SIFT).
4. **Spatial Correspondence**: Two-nearest-neighbor descriptor matching paired with Lowe's ambiguity ratio test.
5. **Projective Planar Geometry**: Estimation of 3x3 homography matrices via Random Sample Consensus (RANSAC) and perspective spatial warping.
6. **Two-View Epipolar Geometry**: Algebraic computation of the rank-2 Fundamental Matrix ($F$) satisfying $x_2^T F x_1 = 0$, inlier correspondence verification, and conjugate epipolar line rendering.
7. **Stereopsis & Depth Recovery**: Dense horizontal disparity estimation (StereoSGBM) and exact metric/relative depth triangulation ($Z = \frac{f \cdot B}{d}$).
8. **Automated Analytics & Reporting**: Non-GUI visualization rendering and structured JSON telemetry export.

---

## 3. Target Users

- **Computer Vision Students & Researchers**: Requiring an interpretable codebase to study classical multi-view geometry and low-level filtering algorithms.
- **Academic Evaluators & Instructors**: Seeking an auditable, verifiable project where every intermediate stage generates reproducible numerical and visual evidence.
- **Embedded & Robotics Engineers**: Developing deterministic visual odometry or depth estimation pipelines for headless or resource-constrained platforms.

---

## 4. High-Level Features

- **Decoupled Architecture**: Each stage exists as an isolated module with typed configuration dataclasses and independent unit tests.
- **Multi-Mode CLI**: Supports granular execution of isolated modules (e.g., `--mode edges`, `--mode geometry`) as well as an end-to-end full pipeline (`--mode full`).
- **Strictly Headless Operation**: Generates high-resolution visualization artifacts directly to disk using the Matplotlib Agg backend, eliminating GUI window lockups on remote servers.
- **Mathematical Transparency**: Avoids synthetic or fabricated metrics; all numbers in the output reports originate from real array computations and statistical analysis.
- **Calibrated vs. Uncalibrated Distinction**: Explicitly flags when depth calculations are physical metric units (meters) versus relative inverse disparities.

---

## 5. Expected Inputs

| Input Type | Supported Formats | Constraints |
| :--- | :--- | :--- |
| **Single Image** | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` | Non-empty, minimum $10 \times 10$ pixels, single-channel or 3-channel RGB. |
| **Stereo Image Pair** | Identical image formats | Left and Right views of the same scene; dimensions harmonized automatically. |
| **Camera Parameters** | Float parameters via CLI | Focal length $f > 0$ (pixels), baseline $B > 0$ (meters). |

---

## 6. Expected Outputs

1. **Processed Visual Artifacts** (saved in designated `--output` directory):
   - `grayscale.png`: Luminance conversion.
   - `enhanced.png`: Multi-panel comparison of original, denoised, CLAHE, and sharpened images.
   - `sobel.png`, `canny.png`: Gradient magnitude and edge maps.
   - `segmentation.png`: Connected components spatial labeling.
   - `sift_keypoints.png`: Scale and orientation keypoint glyphs.
   - `feature_matches.png`: Side-by-side correspondence lines.
   - `homography_warp.png`: Warped perspective and alpha-blended overlay.
   - `epipolar_lines.png`: Conjugate epipolar lines and matching feature points.
   - `disparity.png`: Normalized false-color disparity map.
   - `depth.png`: False-color metric depth map.
2. **Machine-Readable Telemetry Report** (`analysis_report.json`):
   - Execution timestamps, input filenames, image dimensions.
   - Exact algorithmic steps executed.
   - Per-module latency breakdown (milliseconds).
   - Numerical metrics: edge density, keypoint counts, match counts, inlier ratios, mean reprojection errors, disparity limits, depth distributions.

---

## 7. Technical Constraints

- **Python Runtime**: Python 3.10+ (tested on Python 3.12).
- **No Black-Box Deep Learning**: Strictly classical computer vision implementations (OpenCV, NumPy, scikit-image).
- **No GUI Dependency**: No calls to `cv2.imshow()`; headless execution is mandatory.
- **Reproducibility**: All random operations (RANSAC seeds, synthetic data generation) use deterministic random seeds to guarantee repeatable evaluation results.
- **Cross-Platform Compatibility**: Path handling implemented using `pathlib.Path` to ensure seamless execution across Linux, macOS, and Windows.
