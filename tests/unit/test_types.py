"""Tests for foundation types in aquacore.types."""

import torch
import torch.testing

from aquacore import (
    INTERFACE_NORMAL,
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
)


def test_camera_intrinsics_construction(device: torch.device) -> None:
    """CameraIntrinsics stores K and dist_coeffs on the correct device."""
    K = torch.eye(3, dtype=torch.float32, device=device)
    dist_coeffs = torch.zeros(5, dtype=torch.float64, device=device)
    intr = CameraIntrinsics(K=K, dist_coeffs=dist_coeffs, image_size=(1920, 1080))

    assert intr.K.device.type == device.type
    assert intr.dist_coeffs.device.type == device.type
    assert intr.image_size == (1920, 1080)
    assert intr.is_fisheye is False


def test_camera_extrinsics_construction(device: torch.device) -> None:
    """CameraExtrinsics stores R and t on the correct device."""
    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.zeros(3, dtype=torch.float32, device=device)
    extr = CameraExtrinsics(R=R, t=t)

    assert extr.R.device.type == device.type
    assert extr.t.device.type == device.type
    assert extr.R.shape == (3, 3)
    assert extr.t.shape == (3,)


def test_camera_extrinsics_center(device: torch.device) -> None:
    """CameraExtrinsics.C returns -R.T @ t."""
    # Use a 90-degree rotation about Z and a known translation
    R = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    extr = CameraExtrinsics(R=R, t=t)

    expected = -R.T @ t
    torch.testing.assert_close(extr.C, expected, atol=1e-5, rtol=0)


def test_interface_params_defaults() -> None:
    """InterfaceParams has correct default refractive indices."""
    normal = torch.tensor([0.0, 0.0, -1.0])
    params = InterfaceParams(normal=normal, water_z=0.5)

    assert params.n_air == 1.0
    assert params.n_water == 1.333
    assert params.water_z == 0.5


def test_interface_normal_convention() -> None:
    """INTERFACE_NORMAL is [0, 0, -1] per coordinate system convention."""
    expected = torch.tensor([0.0, 0.0, -1.0])
    torch.testing.assert_close(INTERFACE_NORMAL, expected, atol=1e-7, rtol=0)
