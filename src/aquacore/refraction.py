"""Snell's law vector form and ray tracing through refractive interfaces."""

import torch

from .interface import ray_plane_intersection
from .types import InterfaceParams


def snells_law_3d(
    incident_directions: torch.Tensor,
    surface_normal: torch.Tensor,
    n_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Snell's law in 3D vector form to a batch of rays.

    Computes refracted ray directions using the vector form of Snell's law.
    Handles total internal reflection (TIR) by returning a validity mask rather
    than raising an exception or returning NaN.

    Args:
        incident_directions: Incident ray unit vectors, shape (N, 3), float32.
            Must be unit-length (normalized). Direction of travel into the surface.
        surface_normal: Unit normal of the refracting surface, shape (3,), float32.
            Orientation is arbitrary — the function handles both orientations
            internally by checking the sign of dot(incident, normal).
        n_ratio: Ratio of refractive indices n_incident / n_transmitted (float).
            For air-to-water refraction: n_ratio = n_air / n_water ≈ 0.750.
            For water-to-air refraction: n_ratio = n_water / n_air ≈ 1.333.

    Returns:
        directions: Refracted unit direction vectors, shape (N, 3).
            Rows corresponding to TIR cases (valid=False) are set to zeros.
        valid: Boolean mask, shape (N,). False where total internal reflection
            occurs (sin_t_sq > 1.0). True for valid refracted rays.

    Note:
        The function internally orients the surface normal to point from the
        incident medium into the transmission medium. Both air-to-water and
        water-to-air directions are handled correctly without additional setup.
    """
    # cos_i = dot(d, normal) per ray — sign reveals normal orientation
    cos_i = (incident_directions * surface_normal).sum(dim=-1)  # (N,)

    # Orient normal to point from incident medium into transmission medium.
    # When cos_i < 0, the ray travels against the normal (flip it).
    flip = cos_i < 0
    n_oriented = torch.where(flip.unsqueeze(-1), -surface_normal, surface_normal)
    cos_i = cos_i.abs()  # always positive after orientation

    # Snell's law: n1 * sin(θ1) = n2 * sin(θ2) → sin²(θ2) = n_ratio² * sin²(θ1)
    sin_t_sq = n_ratio**2 * (1.0 - cos_i**2)  # (N,)

    # Total internal reflection when sin²(θ_t) > 1
    valid = sin_t_sq <= 1.0  # (N,) bool

    # cos(θ_t) — clamp for numerical stability (sin_t_sq slightly > 1 due to float)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))  # (N,)

    # Vector form of Snell's law:
    # d_t = n_ratio * d_i + (cos_t - n_ratio * cos_i) * n_oriented
    directions = (
        n_ratio * incident_directions
        + (cos_t - n_ratio * cos_i).unsqueeze(-1) * n_oriented
    )  # (N, 3)

    # Normalize (clamp to avoid div-by-zero for TIR rows)
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(min=1e-12)
    directions = directions / norms

    # Zero out TIR directions — caller uses valid mask to filter
    directions = torch.where(
        valid.unsqueeze(-1), directions, torch.zeros_like(directions)
    )

    return directions, valid


def trace_ray_air_to_water(
    origins: torch.Tensor,
    directions: torch.Tensor,
    interface: InterfaceParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace rays from air through a flat water surface (air-to-water refraction).

    Intersects each ray with the water surface plane, then applies Snell's law
    at the intersection point to compute the refracted direction in water.

    Args:
        origins: Ray origins in air (camera centers or arbitrary air points),
            shape (N, 3), float32. World frame.
        directions: Unit ray direction vectors in air, shape (N, 3), float32.
            Must point generally toward the water surface (positive Z component
            for the default +Z-down coordinate system).
        interface: Refractive interface parameters (normal, water_z, n_air, n_water).

    Returns:
        interface_points: Intersection points on the water surface, shape (N, 3).
            Z-coordinate equals interface.water_z. Meaningful only where valid=True.
        refracted_directions: Refracted unit direction vectors in water, shape (N, 3).
            Meaningful only where valid=True.
        valid: Boolean mask, shape (N,). False when the ray is parallel to the
            surface (no intersection), the intersection is behind the origin,
            or total internal reflection occurs at the surface.
    """
    # Intersect ray with horizontal plane z = water_z
    # Plane equation: dot(p, [0,0,1]) = water_z → plane_normal=(0,0,1), plane_d=water_z
    plane_normal = torch.tensor(
        [0.0, 0.0, 1.0], dtype=origins.dtype, device=origins.device
    )
    interface_points, intersect_valid = ray_plane_intersection(
        origins, directions, plane_normal, interface.water_z
    )  # (N, 3), (N,)

    # Apply Snell's law at the interface (air → water)
    n_ratio = interface.n_air / interface.n_water
    refracted_directions, snell_valid = snells_law_3d(
        directions,
        interface.normal.to(device=origins.device, dtype=origins.dtype),
        n_ratio,
    )  # (N, 3), (N,)

    valid = intersect_valid & snell_valid  # (N,)

    return interface_points, refracted_directions, valid


def trace_ray_water_to_air(
    origins: torch.Tensor,
    directions: torch.Tensor,
    interface: InterfaceParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace rays from water through a flat water surface (water-to-air refraction).

    Intersects each ray with the water surface plane, then applies Snell's law
    at the intersection point to compute the refracted direction in air.

    Args:
        origins: Ray origins in water (underwater points), shape (N, 3), float32.
            World frame.
        directions: Unit ray direction vectors in water, shape (N, 3), float32.
            Must point generally toward the water surface (negative Z component
            for the default +Z-down coordinate system).
        interface: Refractive interface parameters (normal, water_z, n_air, n_water).

    Returns:
        interface_points: Intersection points on the water surface, shape (N, 3).
            Z-coordinate equals interface.water_z. Meaningful only where valid=True.
        refracted_directions: Refracted unit direction vectors in air, shape (N, 3).
            Meaningful only where valid=True.
        valid: Boolean mask, shape (N,). False when the ray is parallel to the
            surface, the intersection is behind the origin, or total internal
            reflection occurs (steep underwater angle).
    """
    # Intersect ray with horizontal plane z = water_z
    plane_normal = torch.tensor(
        [0.0, 0.0, 1.0], dtype=origins.dtype, device=origins.device
    )
    interface_points, intersect_valid = ray_plane_intersection(
        origins, directions, plane_normal, interface.water_z
    )  # (N, 3), (N,)

    # Apply Snell's law at the interface (water → air)
    n_ratio = interface.n_water / interface.n_air
    refracted_directions, snell_valid = snells_law_3d(
        directions,
        interface.normal.to(device=origins.device, dtype=origins.dtype),
        n_ratio,
    )  # (N, 3), (N,)

    valid = intersect_valid & snell_valid  # (N,)

    return interface_points, refracted_directions, valid


def refractive_project(
    points: torch.Tensor,
    camera_center: torch.Tensor,
    interface: InterfaceParams,
    n_iterations: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the interface point for underwater 3D points seen by an air-side camera.

    Uses Newton-Raphson iteration (fixed number of steps, no convergence check)
    to locate the point P on the water surface such that the refracted ray from
    the camera through P satisfies Snell's law at P and passes through the
    underwater target point Q.

    Fixed iterations (not convergence-based) ensure deterministic autograd behavior.

    Args:
        points: Underwater 3D points in world frame, shape (N, 3), float32.
            Z-coordinates must be greater than interface.water_z (below surface).
        camera_center: Camera optical center in world frame, shape (3,), float32.
            Must be in air (Z < interface.water_z in +Z-down convention).
        interface: Refractive interface parameters (normal, water_z, n_air, n_water).
        n_iterations: Number of Newton-Raphson iterations (default 10).
            More iterations give better convergence but are more expensive.

    Returns:
        interface_points: Points on the water surface satisfying Snell's law,
            shape (N, 3). Z-coordinate equals interface.water_z.
        valid: Boolean mask, shape (N,). Currently always True — reserved for
            future geometric validity checks (e.g., degenerate configurations).

    Note:
        The returned interface point is on the water surface (z = water_z).
        To get the pixel coordinate, project interface_point through the camera
        model (K @ (R @ interface_point + t)).
    """
    dtype = points.dtype
    device = points.device

    # Camera height above water surface (positive in +Z-down convention: cam_z < water_z)
    h_c = torch.tensor(
        interface.water_z - camera_center[2].item(), dtype=dtype, device=device
    )
    h_c = h_c.clamp(min=1e-12)  # ensure positive

    # Horizontal displacement from camera to each underwater point (XY plane)
    dx = points[:, 0] - camera_center[0]  # (N,)
    dy = points[:, 1] - camera_center[1]  # (N,)
    r_q = torch.sqrt(
        dx * dx + dy * dy + 1e-12
    )  # (N,) horizontal distance, epsilon avoids div0

    # Depth of underwater points below surface (positive in +Z-down: point_z > water_z)
    h_q = points[:, 2] - interface.water_z  # (N,)
    h_q = h_q.clamp(min=1e-12)  # ensure positive

    # Initial guess: linear interpolation (paraxial approximation)
    r_p = r_q * h_c / (h_c + h_q + 1e-12)  # (N,)

    n_air = interface.n_air
    n_water = interface.n_water

    # Fixed Newton-Raphson iterations — no convergence check (preserves autograd graph)
    for _ in range(n_iterations):
        # Air-side geometry (camera to interface point P)
        d_air_sq = r_p * r_p + h_c * h_c
        d_air = torch.sqrt(d_air_sq)  # (N,)

        # Water-side geometry (interface point P to underwater point Q)
        r_diff = r_q - r_p
        d_water_sq = r_diff * r_diff + h_q * h_q
        d_water = torch.sqrt(d_water_sq)  # (N,)

        # Snell's law residual: n_air * sin(θ_air) - n_water * sin(θ_water) = 0
        sin_air = r_p / d_air
        sin_water = r_diff / d_water
        f = n_air * sin_air - n_water * sin_water  # (N,)

        # Derivative of f w.r.t. r_p
        f_prime = n_air * h_c * h_c / (d_air_sq * d_air) + n_water * h_q * h_q / (
            d_water_sq * d_water
        )  # (N,)

        # Newton-Raphson update (epsilon guards div-by-zero; non-in-place for autograd)
        r_p = r_p - f / (f_prime + 1e-12)
        r_p = torch.clamp(r_p, min=0.0)  # must be non-negative
        r_p = torch.minimum(r_p, r_q)  # cannot exceed horizontal range

    # Convert scalar r_p back to 3D interface point
    # Direction from camera to underwater point (horizontal unit vector)
    # r_p is the radial distance from camera projection to interface point
    # The interface point lies along (camera_xy → point_xy) at fraction r_p / r_q
    scale = r_p / r_q  # (N,) — fraction of horizontal displacement

    ix = camera_center[0] + scale * dx  # (N,)
    iy = camera_center[1] + scale * dy  # (N,)
    iz = torch.full((points.shape[0],), interface.water_z, dtype=dtype, device=device)

    interface_points = torch.stack([ix, iy, iz], dim=-1)  # (N, 3)
    valid = torch.ones(points.shape[0], dtype=torch.bool, device=device)

    return interface_points, valid


def refractive_back_project(
    pixel_rays: torch.Tensor,
    camera_centers: torch.Tensor,
    interface: InterfaceParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast refracted rays from pixels through an air-water interface into water.

    Given unit ray directions in air (from camera through pixel), traces each
    ray to the water surface and applies Snell's law to obtain the refracted
    direction in water.

    This is the inverse of refractive_project: refractive_project finds where
    a known underwater point appears on screen; refractive_back_project casts
    a ray from screen into water to search for what lies along that ray.

    Args:
        pixel_rays: Unit ray directions in air (world frame), shape (N, 3), float32.
            Typically computed as (K_inv @ pixel_homogeneous) rotated to world frame.
        camera_centers: Camera optical center(s) in world frame.
            Shape (N, 3) for per-ray camera centers, or (3,) for a single camera
            shared by all rays, float32.
        interface: Refractive interface parameters (normal, water_z, n_air, n_water).

    Returns:
        interface_points: Intersection points on the water surface, shape (N, 3).
            Meaningful only where valid=True.
        water_directions: Refracted unit direction vectors in water, shape (N, 3).
            Meaningful only where valid=True.
        valid: Boolean mask, shape (N,). False when rays are parallel to surface,
            intersection is behind the origin, or total internal reflection occurs.
    """
    # Broadcast scalar camera center to (N, 3) if needed
    if camera_centers.dim() == 1:
        camera_centers = camera_centers.unsqueeze(0).expand(pixel_rays.shape[0], -1)

    return trace_ray_air_to_water(camera_centers, pixel_rays, interface)
