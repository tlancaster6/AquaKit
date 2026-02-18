"""Tests for camera models: pinhole, fisheye, and create_camera factory.

Verifies known-value projection, back-projection, and round-trip correctness
for both models using the device fixture.
"""

from __future__ import annotations

import pytest
import torch

from aquacore import CameraExtrinsics, CameraIntrinsics, create_camera
from aquacore.camera import _FisheyeCamera, _PinholeCamera

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pinhole_intrinsics(device: torch.device) -> CameraIntrinsics:
    """Standard 640x480 pinhole intrinsics with zero distortion."""
    return CameraIntrinsics(
        K=torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        ),
        dist_coeffs=torch.zeros(5, dtype=torch.float64, device=device),
        image_size=(640, 480),
        is_fisheye=False,
    )


def _fisheye_intrinsics(device: torch.device) -> CameraIntrinsics:
    """Standard 640x480 fisheye intrinsics with zero distortion."""
    return CameraIntrinsics(
        K=torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        ),
        dist_coeffs=torch.zeros(4, dtype=torch.float64, device=device),
        image_size=(640, 480),
        is_fisheye=True,
    )


def _identity_extrinsics(device: torch.device) -> CameraExtrinsics:
    """Identity world-to-camera transform."""
    return CameraExtrinsics(
        R=torch.eye(3, dtype=torch.float32, device=device),
        t=torch.zeros(3, dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------------------
# Pinhole tests
# ---------------------------------------------------------------------------


def test_pinhole_project_on_axis(device: torch.device) -> None:
    """On-axis point (0, 0, 5) should project to principal point (320, 240)."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    expected = torch.tensor([[320.0, 240.0]], dtype=torch.float32)
    torch.testing.assert_close(pixels.cpu(), expected, atol=1e-3, rtol=0)
    assert valid.all()


def test_pinhole_project_off_axis(device: torch.device) -> None:
    """Point (1, 0, 5) should project to x = 500*1/5 + 320 = 420, y = 240."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[1.0, 0.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    expected = torch.tensor([[420.0, 240.0]], dtype=torch.float32)
    torch.testing.assert_close(pixels.cpu(), expected, atol=1e-3, rtol=0)
    assert valid.all()


def test_pinhole_behind_camera(device: torch.device) -> None:
    """Point at z < 0 in camera frame should be marked invalid."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, device=device)
    _, valid = cam.project(points)
    assert not valid.any()


def test_pinhole_pixel_to_ray(device: torch.device) -> None:
    """Principal point (320, 240) should back-project to forward ray (0, 0, 1)."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    pixels = torch.tensor([[320.0, 240.0]], dtype=torch.float32, device=device)
    rays = cam.pixel_to_ray(pixels)
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(rays.cpu(), expected, atol=1e-5, rtol=0)


def _stable_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Numerically stable angle between unit vectors using atan2(|cross|, dot).

    Unlike acos(dot), this is numerically stable for small angles where float32
    acos suffers from catastrophic cancellation near 1.0.

    Args:
        a: Unit vectors, shape (N, 3), float32.
        b: Unit vectors, shape (N, 3), float32.

    Returns:
        Angles in radians, shape (N,).
    """
    cross = torch.linalg.cross(a, b)  # (N, 3)
    cross_norm = cross.norm(dim=-1)  # (N,)
    dot = (a * b).sum(dim=-1)  # (N,)
    return torch.atan2(cross_norm, dot)  # (N,)


def test_pinhole_round_trip(device: torch.device) -> None:
    """Project then back-project should recover original ray direction to 1e-5 rad.

    Uses atan2(|cross|, dot) for numerically stable angle measurement since
    acos(dot) suffers from float32 catastrophic cancellation near 1.0.
    """
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[1.0, 2.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    assert valid.all()

    rays = cam.pixel_to_ray(pixels)

    # Expected direction: from camera center (origin) toward the point
    expected_dir = points / points.norm(dim=1, keepdim=True)

    # Stable angle: atan2(|cross|, dot) avoids float32 acos precision loss near 1.0
    angle = _stable_angle(rays.cpu(), expected_dir.cpu())
    assert angle.item() < 1e-5, f"Round-trip angle error: {angle.item():.2e} rad"


def test_pinhole_with_distortion(device: torch.device) -> None:
    """Round-trip with barrel distortion (k1=-0.1) should recover ray to 1e-4 rad."""
    intrinsics = CameraIntrinsics(
        K=torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        ),
        dist_coeffs=torch.tensor(
            [-0.1, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device
        ),
        image_size=(640, 480),
        is_fisheye=False,
    )
    cam = create_camera(intrinsics, _identity_extrinsics(device))
    points = torch.tensor([[1.0, 1.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    assert valid.all()

    rays = cam.pixel_to_ray(pixels)

    expected_dir = points / points.norm(dim=1, keepdim=True)
    angle = _stable_angle(rays.cpu(), expected_dir.cpu())
    assert angle.item() < 1e-4, (
        f"Distorted round-trip angle error: {angle.item():.2e} rad"
    )


# ---------------------------------------------------------------------------
# Fisheye tests
# ---------------------------------------------------------------------------


def test_fisheye_project_on_axis(device: torch.device) -> None:
    """On-axis point (0, 0, 5) should project to principal point (320, 240)."""
    cam = create_camera(_fisheye_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    expected = torch.tensor([[320.0, 240.0]], dtype=torch.float32)
    torch.testing.assert_close(pixels.cpu(), expected, atol=1e-3, rtol=0)
    assert valid.all()


def test_fisheye_pixel_to_ray(device: torch.device) -> None:
    """Principal point (320, 240) should back-project to forward ray (0, 0, 1)."""
    cam = create_camera(_fisheye_intrinsics(device), _identity_extrinsics(device))
    pixels = torch.tensor([[320.0, 240.0]], dtype=torch.float32, device=device)
    rays = cam.pixel_to_ray(pixels)
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(rays.cpu(), expected, atol=1e-5, rtol=0)


def test_fisheye_round_trip(device: torch.device) -> None:
    """Project then back-project should recover original ray direction to 1e-4 rad."""
    cam = create_camera(_fisheye_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[1.0, 2.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    assert valid.all()

    rays = cam.pixel_to_ray(pixels)

    expected_dir = points / points.norm(dim=1, keepdim=True)
    angle = _stable_angle(rays.cpu(), expected_dir.cpu())
    assert angle.item() < 1e-4, (
        f"Fisheye round-trip angle error: {angle.item():.2e} rad"
    )


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_create_camera_dispatches_pinhole(device: torch.device) -> None:
    """create_camera with is_fisheye=False should return a _PinholeCamera."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    assert isinstance(cam, _PinholeCamera)


def test_create_camera_dispatches_fisheye(device: torch.device) -> None:
    """create_camera with is_fisheye=True should return a _FisheyeCamera."""
    cam = create_camera(_fisheye_intrinsics(device), _identity_extrinsics(device))
    assert isinstance(cam, _FisheyeCamera)


def test_create_camera_device_mismatch() -> None:
    """Device mismatch between K and R/t should raise ValueError."""
    cpu_device = torch.device("cpu")
    meta_device = torch.device("meta")

    intrinsics = CameraIntrinsics(
        K=torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=cpu_device,
        ),
        dist_coeffs=torch.zeros(5, dtype=torch.float64, device=cpu_device),
        image_size=(640, 480),
        is_fisheye=False,
    )
    # Put R on "meta" device to force device mismatch
    extrinsics = CameraExtrinsics(
        R=torch.eye(3, dtype=torch.float32, device=meta_device),
        t=torch.zeros(3, dtype=torch.float32, device=cpu_device),
    )

    with pytest.raises(ValueError, match="device"):
        create_camera(intrinsics, extrinsics)


def test_create_camera_bad_K_shape(device: torch.device) -> None:
    """K with wrong shape should raise ValueError."""
    intrinsics = CameraIntrinsics(
        K=torch.zeros(3, 4, dtype=torch.float32, device=device),
        dist_coeffs=torch.zeros(5, dtype=torch.float64),
        image_size=(640, 480),
        is_fisheye=False,
    )
    extrinsics = _identity_extrinsics(device)
    with pytest.raises(ValueError, match="K must have shape"):
        create_camera(intrinsics, extrinsics)


def test_create_camera_bad_t_shape(device: torch.device) -> None:
    """t with wrong shape should raise ValueError."""
    intrinsics = _pinhole_intrinsics(device)
    extrinsics = CameraExtrinsics(
        R=torch.eye(3, dtype=torch.float32, device=device),
        t=torch.zeros(3, 1, dtype=torch.float32, device=device),
    )
    with pytest.raises(ValueError, match="t must have shape"):
        create_camera(intrinsics, extrinsics)


def test_pinhole_batch(device: torch.device) -> None:
    """project() and pixel_to_ray() must handle batches correctly."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    # 3 different on-axis points at different depths
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 10.0]],
        dtype=torch.float32,
        device=device,
    )
    pixels, valid = cam.project(points)
    assert pixels.shape == (3, 2)
    assert valid.shape == (3,)
    assert valid.all()

    rays = cam.pixel_to_ray(pixels)
    assert rays.shape == (3, 3)
    # All should be forward rays
    expected = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    torch.testing.assert_close(rays.cpu(), expected, atol=1e-5, rtol=0)


def test_pinhole_off_axis_y(device: torch.device) -> None:
    """Point (0, 1, 5) should project to y = 500*1/5 + 240 = 340, x = 320."""
    cam = create_camera(_pinhole_intrinsics(device), _identity_extrinsics(device))
    points = torch.tensor([[0.0, 1.0, 5.0]], dtype=torch.float32, device=device)
    pixels, valid = cam.project(points)
    expected = torch.tensor([[320.0, 340.0]], dtype=torch.float32)
    torch.testing.assert_close(pixels.cpu(), expected, atol=1e-3, rtol=0)
    assert valid.all()
