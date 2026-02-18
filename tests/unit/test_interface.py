"""Tests for ray-plane intersection in aquacore.interface."""

import torch
import torch.testing

from aquacore import ray_plane_intersection


def test_ray_hits_plane(device: torch.device) -> None:
    """Ray from (0,0,-1) pointing +Z hits the plane at z=0, intersection at (0,0,0)."""
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    # Plane: dot(p, [0,0,1]) = 0  (z = 0)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    plane_d = 0.0

    points, valid = ray_plane_intersection(origins, directions, plane_normal, plane_d)

    assert valid[0].item() is True
    expected = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    torch.testing.assert_close(points, expected, atol=1e-5, rtol=0)


def test_ray_parallel_to_plane(device: torch.device) -> None:
    """Ray direction parallel to the plane returns valid=False."""
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)
    # Direction is [1, 0, 0] which is perpendicular to normal [0, 0, 1]
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    plane_d = 0.0

    _, valid = ray_plane_intersection(origins, directions, plane_normal, plane_d)

    assert valid[0].item() is False


def test_ray_behind_origin(device: torch.device) -> None:
    """Ray pointing away from the plane (t < 0) returns valid=False."""
    # Ray starts at z=1 and points in +Z direction (away from z=0 plane)
    origins = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    plane_d = 0.0  # plane at z=0

    _, valid = ray_plane_intersection(origins, directions, plane_normal, plane_d)

    assert valid[0].item() is False


def test_batched_intersection(device: torch.device) -> None:
    """Batch of 5 rays with mixed valid/invalid cases."""
    # Plane: z = 1.0 (dot(p, [0,0,1]) = 1)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    plane_d = 1.0

    origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # ray 0: from z=0, pointing +Z -> hits at (0,0,1)
            [1.0, 0.0, 0.0],  # ray 1: from x=1, pointing +Z -> hits at (1,0,1)
            [0.0, 0.0, 0.0],  # ray 2: from z=0, pointing -Z -> t < 0, invalid
            [0.0, 1.0, 0.0],  # ray 3: from y=1, dir parallel to plane -> invalid
            [
                0.5,
                0.5,
                0.0,
            ],  # ray 4: diagonal origin, pointing +Z -> hits at (0.5,0.5,1)
        ],
        dtype=torch.float32,
        device=device,
    )
    directions = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # ray 0: +Z
            [0.0, 0.0, 1.0],  # ray 1: +Z
            [0.0, 0.0, -1.0],  # ray 2: -Z (behind)
            [1.0, 0.0, 0.0],  # ray 3: +X (parallel to plane)
            [0.0, 0.0, 1.0],  # ray 4: +Z
        ],
        dtype=torch.float32,
        device=device,
    )

    points, valid = ray_plane_intersection(origins, directions, plane_normal, plane_d)

    assert valid.shape == (5,)
    assert points.shape == (5, 3)

    expected_valid = torch.tensor([True, True, False, False, True], device=device)
    assert (valid == expected_valid).all()

    # Check valid intersection points
    torch.testing.assert_close(
        points[0],
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        points[1],
        torch.tensor([1.0, 0.0, 1.0], device=device, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        points[4],
        torch.tensor([0.5, 0.5, 1.0], device=device, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )
