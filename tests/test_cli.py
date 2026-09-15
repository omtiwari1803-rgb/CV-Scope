"""
Integration tests for CV-Scope CLI commands, execution modes, and error handling.
"""

import sys
import subprocess
from pathlib import Path
import pytest

MAIN_SCRIPT = Path(__file__).resolve().parent.parent / "main.py"
SAMPLE_SINGLE = Path(__file__).resolve().parent.parent / "data" / "single" / "sample.png"
SAMPLE_LEFT = Path(__file__).resolve().parent.parent / "data" / "stereo" / "left.png"
SAMPLE_RIGHT = Path(__file__).resolve().parent.parent / "data" / "stereo" / "right.png"


def run_cli_command(args: list) -> subprocess.CompletedProcess:
    """Run main.py with the specified arguments using the current Python executable."""
    cmd = [sys.executable, str(MAIN_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_help():
    """Verify --help flag exits cleanly with returncode 0 and prints usage."""
    result = run_cli_command(["--help"])
    assert result.returncode == 0
    assert "CV-Scope" in result.stdout
    assert "--mode" in result.stdout


def test_cli_enhance_mode(tmp_path):
    """Verify enhance mode runs and outputs artifacts."""
    out_dir = tmp_path / "enhance_out"
    result = run_cli_command([
        "--mode", "enhance",
        "--input", str(SAMPLE_SINGLE),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "enhanced.png").exists()
    assert (out_dir / "analysis_report.json").exists()


def test_cli_edges_mode(tmp_path):
    """Verify edges mode runs and outputs artifacts."""
    out_dir = tmp_path / "edges_out"
    result = run_cli_command([
        "--mode", "edges",
        "--input", str(SAMPLE_SINGLE),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "canny.png").exists()
    assert (out_dir / "sobel.png").exists()
    assert (out_dir / "analysis_report.json").exists()


def test_cli_features_mode(tmp_path):
    """Verify features mode runs and outputs keypoints image."""
    out_dir = tmp_path / "features_out"
    result = run_cli_command([
        "--mode", "features",
        "--input", str(SAMPLE_SINGLE),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "sift_keypoints.png").exists()


def test_cli_match_mode(tmp_path):
    """Verify match mode finds matches between stereo pair."""
    out_dir = tmp_path / "match_out"
    result = run_cli_command([
        "--mode", "match",
        "--left", str(SAMPLE_LEFT),
        "--right", str(SAMPLE_RIGHT),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "feature_matches.png").exists()


def test_cli_geometry_mode(tmp_path):
    """Verify geometry mode computes homography."""
    out_dir = tmp_path / "geometry_out"
    result = run_cli_command([
        "--mode", "geometry",
        "--left", str(SAMPLE_LEFT),
        "--right", str(SAMPLE_RIGHT),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "homography_warp.png").exists()


def test_cli_epipolar_mode(tmp_path):
    """Verify epipolar mode computes fundamental matrix and renders epipolar lines."""
    out_dir = tmp_path / "epipolar_out"
    result = run_cli_command([
        "--mode", "epipolar",
        "--left", str(SAMPLE_LEFT),
        "--right", str(SAMPLE_RIGHT),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "epipolar_lines.png").exists()


def test_cli_depth_mode(tmp_path):
    """Verify depth mode computes disparity and depth maps."""
    out_dir = tmp_path / "depth_out"
    result = run_cli_command([
        "--mode", "depth",
        "--left", str(SAMPLE_LEFT),
        "--right", str(SAMPLE_RIGHT),
        "--output", str(out_dir)
    ])
    assert result.returncode == 0
    assert (out_dir / "disparity.png").exists()
    assert (out_dir / "depth.png").exists()


def test_cli_missing_file():
    """Verify missing input file yields non-zero exit code with error message."""
    result = run_cli_command([
        "--mode", "enhance",
        "--input", "non_existent_file.png"
    ])
    assert result.returncode != 0
    assert "File Not Found" in result.stderr or "File Not Found" in result.stdout or "not found" in result.stderr.lower()


def test_cli_missing_stereo_arg():
    """Verify providing only --left for stereo mode yields validation error."""
    result = run_cli_command([
        "--mode", "depth",
        "--left", str(SAMPLE_LEFT)
    ])
    assert result.returncode != 0
    assert "requires both '--left <path>' and '--right <path>'" in result.stderr or "requires both" in result.stdout
