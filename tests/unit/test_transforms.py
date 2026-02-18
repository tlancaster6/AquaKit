"""Tests for rotation and pose transform utilities in aquacore.transforms."""

import math

import torch
import torch.testing

from aquacore import (
    camera_center,
    compose_poses,
    invert_pose,
    matrix_to_rvec,
    rvec_to_matrix,
)


def test_rvec_to_matrix_zero(device: torch.device) -> None:
    """Zero rotation vector produces identity matrix."""
    rvec = torch.zeros(3, dtype=torch.float32, device=device)
    R = rvec_to_matrix(rvec)
    expected = torch.eye(3, dtype=torch.float32, device=device)
    torch.testing.assert_close(R, expected, atol=1e-6, rtol=0)


def test_rvec_to_matrix_90_z(device: torch.device) -> None:
    """90-degree rotation about Z: maps [1,0,0] to [0,1,0]."""
    rvec = torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float32, device=device)
    R = rvec_to_matrix(rvec)

    # R @ [1, 0, 0] should give approximately [0, 1, 0]
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    rotated = R @ x_axis
    expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(rotated, expected, atol=1e-5, rtol=0)


def test_rvec_to_matrix_180(device: torch.device) -> None:
    """180-degree rotation about X: maps [0,1,0] to [0,-1,0]. Tests theta=pi edge case."""
    rvec = torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float32, device=device)
    R = rvec_to_matrix(rvec)

    # R @ [0, 1, 0] should give approximately [0, -1, 0]
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    rotated = R @ y_axis
    expected = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(rotated, expected, atol=1e-5, rtol=0)


def test_matrix_to_rvec_roundtrip(device: torch.device) -> None:
    """rvec -> matrix -> rvec roundtrip preserves angle and axis direction."""
    # Use a non-degenerate rotation (avoid 0 and pi)
    rvec_in = torch.tensor([0.3, -0.7, 0.5], dtype=torch.float32, device=device)
    R = rvec_to_matrix(rvec_in)
    rvec_out = matrix_to_rvec(R)

    # The rotation angle (norm) must match
    angle_in = torch.linalg.norm(rvec_in)
    angle_out = torch.linalg.norm(rvec_out)
    torch.testing.assert_close(angle_out, angle_in, atol=1e-5, rtol=0)

    # The axis must match in direction (or be negated for equivalent rotation)
    axis_in = rvec_in / angle_in
    axis_out = rvec_out / angle_out
    # axis_out should equal axis_in (same direction for angle in (0, pi))
    torch.testing.assert_close(axis_out, axis_in, atol=1e-5, rtol=0)


def test_compose_poses_with_identity(device: torch.device) -> None:
    """Composing identity pose with a known pose returns the known pose."""
    R_id = torch.eye(3, dtype=torch.float32, device=device)
    t_id = torch.zeros(3, dtype=torch.float32, device=device)

    R2 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    R_out, t_out = compose_poses(R_id, t_id, R2, t2)

    torch.testing.assert_close(R_out, R2, atol=1e-6, rtol=0)
    torch.testing.assert_close(t_out, t2, atol=1e-6, rtol=0)


def test_compose_poses_known_values(device: torch.device) -> None:
    """Composing two non-trivial poses gives the expected result."""
    # Pose 1: 90-degree rotation about Z, translation [1, 0, 0]
    R1 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Pose 2: 90-degree rotation about Z, translation [0, 1, 0]
    R2 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    R_out, t_out = compose_poses(R1, t1, R2, t2)

    # R_out should be 180-degree rotation about Z: R2 @ R1
    R_expected = R2 @ R1
    # t_out should be R2 @ t1 + t2
    t_expected = R2 @ t1 + t2

    torch.testing.assert_close(R_out, R_expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(t_out, t_expected, atol=1e-6, rtol=0)


def test_invert_pose(device: torch.device) -> None:
    """Inverting a pose then composing with original yields identity."""
    R = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    R_inv, t_inv = invert_pose(R, t)
    R_composed, t_composed = compose_poses(R, t, R_inv, t_inv)

    torch.testing.assert_close(
        R_composed, torch.eye(3, device=device, dtype=torch.float32), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        t_composed,
        torch.zeros(3, device=device, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )


def test_camera_center(device: torch.device) -> None:
    """camera_center(R, t) matches -R.T @ t."""
    R = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    C = camera_center(R, t)
    expected = -R.T @ t
    torch.testing.assert_close(C, expected, atol=1e-5, rtol=0)
