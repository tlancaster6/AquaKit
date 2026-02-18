"""Batched ray intersection and point-to-ray distance computation."""

import torch


def triangulate_rays(
    rays: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Triangulate a 3D point from two or more rays using a closed-form linear solve.

    Finds the point P that minimizes the sum of squared perpendicular distances
    to all rays. This is the standard DLT-style midpoint triangulation algorithm:

        A = sum_i (I - d_i @ d_i.T)
        b = sum_i (I - d_i @ d_i.T) @ o_i
        P = solve(A, b)

    where o_i and d_i are the origin and unit direction of the i-th ray.

    The device and dtype are inferred from the first ray's origin tensor. All
    ray tensors must be on the same device.

    Args:
        rays: List of (origin, direction) tuples, each containing (3,) tensors.
            At least two non-parallel rays are required. Directions need not be
            pre-normalized — they are normalized internally.

    Returns:
        point: Triangulated 3D point, shape (3,), same device and dtype as inputs.

    Raises:
        ValueError: If the ray configuration is degenerate (e.g., all rays are
            parallel), making the linear system singular.

    Note:
        For refractive triangulation (TRI-03), use the refracted ray origins
        (on the water surface) and refracted directions (in water) as inputs.
        The algorithm itself is geometry-agnostic.
    """
    device = rays[0][0].device
    dtype = rays[0][0].dtype

    a_sum = torch.zeros(3, 3, device=device, dtype=dtype)
    b_sum = torch.zeros(3, device=device, dtype=dtype)

    for origin, direction in rays:
        # Normalize direction (allow unnormalized inputs for robustness)
        d = direction / torch.linalg.norm(direction)

        # Projection matrix onto the plane perpendicular to d: I - d @ d.T
        i_minus_ddt = torch.eye(3, device=device, dtype=dtype) - torch.outer(d, d)

        a_sum = a_sum + i_minus_ddt
        b_sum = b_sum + i_minus_ddt @ origin

    try:
        point = torch.linalg.solve(a_sum, b_sum)
    except torch.linalg.LinAlgError as err:  # type: ignore[attr-defined]
        raise ValueError("Degenerate ray configuration") from err

    return point


def point_to_ray_distance(
    point: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
) -> torch.Tensor:
    """Compute the perpendicular distance from a 3D point to a ray.

    Measures how far the point is from the closest point on the ray (i.e., the
    component of the offset vector orthogonal to the ray direction). This is
    used to evaluate triangulation reprojection error.

    Args:
        point: 3D query point, shape (3,), float32.
        ray_origin: Ray origin, shape (3,), float32.
        ray_direction: Unit direction vector of the ray, shape (3,), float32.
            Must be normalized; if not, results will be scaled accordingly.

    Returns:
        distance: Scalar tensor — perpendicular distance from point to ray.
            Non-negative. Zero if and only if the point lies exactly on the ray.

    Formula:
        v = point - ray_origin
        distance = ||v - (v · d) * d||
        where d = ray_direction (assumed unit-length)
    """
    v = point - ray_origin
    # Project v onto ray direction, then subtract to get perpendicular component
    proj = (v * ray_direction).sum() * ray_direction
    return torch.linalg.norm(v - proj)
