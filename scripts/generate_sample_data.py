"""
Script to generate mathematically rigorous, high-contrast, royalty-free sample datasets
for CV-Scope evaluation:
1. data/single/sample.png: Geometric textures, edges, gradients, and multiple contrast levels.
2. data/stereo/left.png & right.png: Ground-truth planar 3D scene projected onto stereo pair
   with known depth planes and horizontal parallax disparity.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def generate_single_sample(output_path: Path, width: int = 640, height: int = 480) -> None:
    """Generate high-contrast textured image with distinct shapes and gradients."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Background gradient
    for y in range(height):
        grad_val = int(40 + 80 * (y / height))
        img[y, :] = [grad_val, grad_val, grad_val + 20]

    # 2. Checkerboard texture block in top-left
    cb_size = 20
    for y in range(30, 190, cb_size):
        for x in range(30, 190, cb_size):
            if ((x // cb_size) + (y // cb_size)) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + cb_size, y + cb_size), (210, 210, 210), -1)
            else:
                cv2.rectangle(img, (x, y), (x + cb_size, y + cb_size), (40, 60, 80), -1)

    # 3. Concentric rings in top-right (rich edge & gradient profile)
    center_ring = (480, 110)
    for r in range(80, 10, -14):
        color = (255, 180, 50) if (r // 14) % 2 == 0 else (30, 30, 30)
        cv2.circle(img, center_ring, r, color, 3)

    # 4. Textured Polygon / Star in center
    pts = np.array([
        [320, 160], [350, 230], [430, 240], [370, 290],
        [390, 370], [320, 330], [250, 370], [270, 290],
        [210, 240], [290, 230]
    ], np.int32)
    cv2.fillPoly(img, [pts], (70, 170, 240))
    cv2.polylines(img, [pts], True, (255, 255, 255), 3)

    # 5. Distinct high-contrast geometric objects at bottom
    # Red rectangle
    cv2.rectangle(img, (50, 300), (180, 420), (50, 50, 220), -1)
    cv2.rectangle(img, (50, 300), (180, 420), (255, 255, 255), 2)

    # Green circle
    cv2.circle(img, (480, 360), 65, (50, 200, 50), -1)
    cv2.circle(img, (480, 360), 65, (255, 255, 255), 2)

    # 6. Fine high-frequency grid pattern in bottom center for SIFT & edge detection
    for i in range(250, 390, 10):
        cv2.line(img, (i, 380), (i + 15, 450), (220, 220, 220), 1)
        cv2.line(img, (390 - (i - 250), 380), (390 - (i - 250) - 15, 450), (180, 180, 240), 1)

    # 7. Add subtle Gaussian noise to simulate real sensor acquisition
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as RGB for consistency with utils.save_image
    cv2.imwrite(str(output_path), cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    print(f"Generated single sample image: {output_path} ({width}x{height})")


def generate_stereo_pair(left_path: Path, right_path: Path, width: int = 640, height: int = 480) -> None:
    """
    Generate a mathematically consistent synthetic stereo image pair.

    Camera setup:
    - Focal length f = 800.0 pixels
    - Stereo baseline B = 0.1 meters
    - Background plane at depth Z_bg = 2.5m -> disparity d_bg = (800 * 0.1) / 2.5 = 32 pixels
    - Middle object plane at depth Z_mid = 1.33m -> disparity d_mid = 80 / 1.33 = 60 pixels
    - Foreground object at depth Z_fg = 0.8m -> disparity d_fg = 80 / 0.8 = 100 pixels
    """
    # Create textured background scene
    # Using deterministic pseudorandom texture so features are dense and trackable
    rng = np.random.default_rng(seed=123)

    # Base background texture (canvas larger than width to allow shift)
    canvas_w = width + 150
    bg_canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)

    # Fill background with distinct textured patterns
    for y in range(0, height, 16):
        for x in range(0, canvas_w, 16):
            val = int(rng.integers(60, 180))
            color = [val, int(val * 0.9), int(val * 1.1)]
            bg_canvas[y:y+16, x:x+16] = color

    # Add texturing and grid lines to background
    for i in range(0, canvas_w, 32):
        cv2.line(bg_canvas, (i, 0), (i, height), (40, 40, 40), 1)
    for j in range(0, height, 32):
        cv2.line(bg_canvas, (0, j), (canvas_w, j), (40, 40, 40), 1)

    # Disparities (horizontal shifts in right image: x_right = x_left - disparity)
    d_bg = 16   # Background disparity
    d_mid = 36  # Middle tier disparity
    d_fg = 64   # Foreground tier disparity

    # Left image view
    img_left = bg_canvas[:, 80:80 + width].copy()

    # Right image view: background shifted by d_bg
    img_right = bg_canvas[:, (80 - d_bg):(80 - d_bg) + width].copy()

    # Layer 2: Middle tier objects (e.g. geometric blocks)
    # Object A: Textured box at [x=100..240, y=140..340]
    box_w, box_h = 140, 200
    mid_tex = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    for y in range(0, box_h, 10):
        for x in range(0, box_w, 10):
            c = (220, 120, 50) if ((x // 10) + (y // 10)) % 2 == 0 else (180, 80, 30)
            mid_tex[y:y+10, x:x+10] = c
    cv2.rectangle(mid_tex, (0, 0), (box_w - 1, box_h - 1), (255, 255, 255), 2)

    # Place in left image
    x_l_mid, y_mid = 100, 140
    img_left[y_mid:y_mid+box_h, x_l_mid:x_l_mid+box_w] = mid_tex

    # Place in right image with disparity d_mid
    x_r_mid = x_l_mid - d_mid
    img_right[y_mid:y_mid+box_h, x_r_mid:x_r_mid+box_w] = mid_tex

    # Layer 3: Foreground tier object (e.g. circle / polygon)
    # Circle at center-right
    cx_l, cy_l, radius = 460, 250, 70
    # Left view foreground circle
    cv2.circle(img_left, (cx_l, cy_l), radius, (40, 210, 120), -1)
    cv2.circle(img_left, (cx_l, cy_l), radius, (255, 255, 255), 3)
    # Add internal crosshair / marker inside circle for sharp SIFT feature points
    cv2.line(img_left, (cx_l - 30, cy_l), (cx_l + 30, cy_l), (0, 0, 0), 2)
    cv2.line(img_left, (cx_l, cy_l - 30), (cx_l, cy_l + 30), (0, 0, 0), 2)

    # Right view foreground circle with disparity d_fg
    cx_r = cx_l - d_fg
    cv2.circle(img_right, (cx_r, cy_l), radius, (40, 210, 120), -1)
    cv2.circle(img_right, (cx_r, cy_l), radius, (255, 255, 255), 3)
    cv2.line(img_right, (cx_r - 30, cy_l), (cx_r + 30, cy_l), (0, 0, 0), 2)
    cv2.line(img_right, (cx_r, cy_l - 30), (cx_r, cy_l + 30), (0, 0, 0), 2)

    # Add sensor noise
    noise_l = rng.normal(0, 2.0, img_left.shape).astype(np.float32)
    noise_r = rng.normal(0, 2.0, img_right.shape).astype(np.float32)
    img_left = np.clip(img_left.astype(np.float32) + noise_l, 0, 255).astype(np.uint8)
    img_right = np.clip(img_right.astype(np.float32) + noise_r, 0, 255).astype(np.uint8)

    left_path.parent.mkdir(parents=True, exist_ok=True)
    right_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(left_path), cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
    print(f"Generated stereo pair: {left_path} and {right_path} ({width}x{height})")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    single_out = base_dir / "data" / "single" / "sample.png"
    stereo_left = base_dir / "data" / "stereo" / "left.png"
    stereo_right = base_dir / "data" / "stereo" / "right.png"

    generate_single_sample(single_out)
    generate_stereo_pair(stereo_left, stereo_right)
