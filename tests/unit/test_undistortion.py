"""Tests for undistortion.py: compute_undistortion_maps and undistort_image."""

import numpy as np
import pytest
import torch

from aquacore import compute_undistortion_maps, undistort_image
from aquacore.calibration import CameraData
from aquacore.types import CameraExtrinsics, CameraIntrinsics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_IMAGE_SIZE = (640, 480)  # (width, height)
_W, _H = _IMAGE_SIZE


@pytest.fixture
def pinhole_camera() -> CameraData:
    """Standard pinhole CameraData with non-trivial distortion coefficients."""
    return CameraData(
        name="test_pinhole",
        intrinsics=CameraIntrinsics(
            K=torch.tensor(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            dist_coeffs=torch.tensor(
                [0.1, -0.2, 0.001, 0.002, 0.03], dtype=torch.float64
            ),
            image_size=_IMAGE_SIZE,
            is_fisheye=False,
        ),
        extrinsics=CameraExtrinsics(
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
        ),
        is_auxiliary=False,
    )


@pytest.fixture
def fisheye_camera() -> CameraData:
    """Fisheye CameraData with 4 equidistant distortion coefficients."""
    return CameraData(
        name="test_fisheye",
        intrinsics=CameraIntrinsics(
            K=torch.tensor(
                [[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            dist_coeffs=torch.tensor([0.05, -0.01, 0.003, -0.001], dtype=torch.float64),
            image_size=_IMAGE_SIZE,
            is_fisheye=True,
        ),
        extrinsics=CameraExtrinsics(
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
        ),
        is_auxiliary=False,
    )


@pytest.fixture
def rgb_image() -> torch.Tensor:
    """Synthetic (H, W, 3) uint8 image tensor."""
    return torch.randint(0, 255, (_H, _W, 3), dtype=torch.uint8)


@pytest.fixture
def gray_image() -> torch.Tensor:
    """Synthetic (H, W) uint8 image tensor."""
    return torch.randint(0, 255, (_H, _W), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# compute_undistortion_maps — pinhole
# ---------------------------------------------------------------------------


def test_pinhole_maps_shape(pinhole_camera: CameraData) -> None:
    """Pinhole maps have shape (H, W)."""
    map_x, map_y = compute_undistortion_maps(pinhole_camera)
    assert map_x.shape == (_H, _W), f"map_x shape {map_x.shape} != ({_H}, {_W})"
    assert map_y.shape == (_H, _W), f"map_y shape {map_y.shape} != ({_H}, {_W})"


def test_pinhole_maps_dtype(pinhole_camera: CameraData) -> None:
    """Pinhole maps are np.float32 arrays."""
    map_x, map_y = compute_undistortion_maps(pinhole_camera)
    assert map_x.dtype == np.float32, f"map_x dtype {map_x.dtype} != float32"
    assert map_y.dtype == np.float32, f"map_y dtype {map_y.dtype} != float32"


# ---------------------------------------------------------------------------
# compute_undistortion_maps — fisheye
# ---------------------------------------------------------------------------


def test_fisheye_maps_shape(fisheye_camera: CameraData) -> None:
    """Fisheye maps have shape (H, W)."""
    map_x, map_y = compute_undistortion_maps(fisheye_camera)
    assert map_x.shape == (_H, _W), f"map_x shape {map_x.shape} != ({_H}, {_W})"
    assert map_y.shape == (_H, _W), f"map_y shape {map_y.shape} != ({_H}, {_W})"


def test_fisheye_maps_dtype(fisheye_camera: CameraData) -> None:
    """Fisheye maps are np.float32 arrays."""
    map_x, map_y = compute_undistortion_maps(fisheye_camera)
    assert map_x.dtype == np.float32, f"map_x dtype {map_x.dtype} != float32"
    assert map_y.dtype == np.float32, f"map_y dtype {map_y.dtype} != float32"


# ---------------------------------------------------------------------------
# compute_undistortion_maps — return type
# ---------------------------------------------------------------------------


def test_returns_tuple(pinhole_camera: CameraData) -> None:
    """Return value is a tuple of length 2."""
    result = compute_undistortion_maps(pinhole_camera)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected tuple length 2, got {len(result)}"


def test_maps_contain_valid_coordinates(pinhole_camera: CameraData) -> None:
    """Most map coordinate values are within valid pixel bounds (not all NaN/inf)."""
    map_x, map_y = compute_undistortion_maps(pinhole_camera)
    # No NaN or inf
    assert not np.any(np.isnan(map_x)), "map_x contains NaN"
    assert not np.any(np.isnan(map_y)), "map_y contains NaN"
    assert not np.any(np.isinf(map_x)), "map_x contains inf"
    assert not np.any(np.isinf(map_y)), "map_y contains inf"
    # Most values should be in valid pixel range (allow some border overshoot)
    frac_x_valid = np.mean((map_x >= -_W) & (map_x <= 2 * _W))
    frac_y_valid = np.mean((map_y >= -_H) & (map_y <= 2 * _H))
    assert frac_x_valid > 0.8, f"Only {frac_x_valid:.1%} of map_x within range"
    assert frac_y_valid > 0.8, f"Only {frac_y_valid:.1%} of map_y within range"


# ---------------------------------------------------------------------------
# undistort_image
# ---------------------------------------------------------------------------


def test_output_shape_matches_input(
    pinhole_camera: CameraData, rgb_image: torch.Tensor
) -> None:
    """Output shape matches input shape (H, W, 3)."""
    maps = compute_undistortion_maps(pinhole_camera)
    out = undistort_image(rgb_image, maps)
    assert out.shape == rgb_image.shape, (
        f"Shape mismatch: {out.shape} != {rgb_image.shape}"
    )


def test_output_dtype_uint8(
    pinhole_camera: CameraData, rgb_image: torch.Tensor
) -> None:
    """uint8 input produces uint8 output."""
    maps = compute_undistortion_maps(pinhole_camera)
    out = undistort_image(rgb_image, maps)
    assert out.dtype == torch.uint8, f"dtype {out.dtype} != uint8"


def test_output_device_matches_input(
    pinhole_camera: CameraData, rgb_image: torch.Tensor
) -> None:
    """Output tensor is on the same device as the input tensor."""
    maps = compute_undistortion_maps(pinhole_camera)
    out = undistort_image(rgb_image, maps)
    assert out.device == rgb_image.device, (
        f"Device mismatch: output {out.device} != input {rgb_image.device}"
    )


def test_grayscale_image(pinhole_camera: CameraData, gray_image: torch.Tensor) -> None:
    """Single-channel (H, W) image is undistorted correctly."""
    maps = compute_undistortion_maps(pinhole_camera)
    out = undistort_image(gray_image, maps)
    assert out.shape == gray_image.shape, (
        f"Shape mismatch: {out.shape} != {gray_image.shape}"
    )
    assert out.dtype == torch.uint8


def test_identity_distortion(rgb_image: torch.Tensor) -> None:
    """Zero distortion coefficients: undistorted image matches original."""
    zero_dist_camera = CameraData(
        name="test_zero_dist",
        intrinsics=CameraIntrinsics(
            K=torch.tensor(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            dist_coeffs=torch.zeros(5, dtype=torch.float64),
            image_size=_IMAGE_SIZE,
            is_fisheye=False,
        ),
        extrinsics=CameraExtrinsics(
            R=torch.eye(3, dtype=torch.float32),
            t=torch.zeros(3, dtype=torch.float32),
        ),
        is_auxiliary=False,
    )
    maps = compute_undistortion_maps(zero_dist_camera)
    out = undistort_image(rgb_image, maps)
    # With zero distortion and alpha=0, the interior should match closely.
    # Allow for bilinear interpolation rounding; compare pixels in center crop.
    orig_np = rgb_image.numpy().astype(np.int32)
    out_np = out.numpy().astype(np.int32)
    # Use a centre crop to avoid border effects from remap
    margin = 40
    orig_crop = orig_np[margin : _H - margin, margin : _W - margin]
    out_crop = out_np[margin : _H - margin, margin : _W - margin]
    max_diff = np.max(np.abs(orig_crop - out_crop))
    assert max_diff <= 2, (
        f"Max pixel diff {max_diff} exceeds tolerance 2 for zero distortion"
    )


# ---------------------------------------------------------------------------
# Integration tests (no files, purely synthetic)
# ---------------------------------------------------------------------------


def test_roundtrip_pinhole(pinhole_camera: CameraData) -> None:
    """Pinhole round-trip: compute maps, undistort synthetic image, check shape/device."""
    image = torch.randint(0, 255, (_H, _W, 3), dtype=torch.uint8)
    maps = compute_undistortion_maps(pinhole_camera)
    out = undistort_image(image, maps)
    assert out.shape == image.shape
    assert out.device == image.device
    assert out.dtype == torch.uint8


def test_roundtrip_fisheye(fisheye_camera: CameraData) -> None:
    """Fisheye round-trip: compute maps, undistort synthetic image, check shape/device."""
    image = torch.randint(0, 255, (_H, _W, 3), dtype=torch.uint8)
    maps = compute_undistortion_maps(fisheye_camera)
    out = undistort_image(image, maps)
    assert out.shape == image.shape
    assert out.device == image.device
    assert out.dtype == torch.uint8
