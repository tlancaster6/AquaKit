"""Refractive projection model wrapping Phase 1 physics primitives."""

from typing import Any

import torch

from ..refraction import refractive_project, snells_law_3d
from ..types import InterfaceParams


class RefractiveProjectionModel:
    """Stateful per-camera refractive projection model.

    Wraps Phase 1's Newton-Raphson and Snell's law primitives into a
    convenient camera object. Holds camera intrinsics (K), extrinsics (R, t),
    and interface parameters (water_z, normal, n_air, n_water). Derived
    quantities (K_inv, camera center C, n_ratio) are precomputed at
    construction time to avoid repeated computation.

    The coordinate system follows the AquaKit convention: world origin at
    reference camera optical center, +X right, +Y forward, +Z down into
    water. Extrinsics: ``p_cam = R @ p_world + t``.

    Note:
        All tensor attributes must be on the same device. Use ``.to(device)``
        to transfer the model. Input tensors passed to ``project()`` and
        ``back_project()`` must also be on the same device — no silent
        device transfer is performed.
    """

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        water_z: float,
        normal: torch.Tensor,
        n_air: float,
        n_water: float,
    ) -> None:
        """Construct a refractive projection model from raw tensors.

        Args:
            K: Camera intrinsic matrix, shape (3, 3), float32.
            R: Rotation matrix (world to camera), shape (3, 3), float32.
            t: Translation vector (world to camera), shape (3,), float32.
                Transform: p_cam = R @ p_world + t.
            water_z: Z-coordinate of water surface in world frame (meters).
                In the +Z-down convention, water_z is positive (surface is
                below the camera).
            normal: Unit normal of the water surface from water toward air,
                shape (3,), float32. Typically [0, 0, -1].
            n_air: Refractive index of air (typically 1.0).
            n_water: Refractive index of water (typically 1.333).
        """
        self.K = K
        self.R = R
        self.t = t
        self.water_z = water_z
        self.normal = normal
        self.n_air = n_air
        self.n_water = n_water
        # Precomputed derived quantities
        self.K_inv = torch.linalg.inv(K)  # (3, 3)
        self.C = -R.T @ t  # (3,) camera center in world frame
        self.n_ratio = n_air / n_water  # scalar float

    @classmethod
    def from_camera(
        cls,
        camera: Any,
        interface: InterfaceParams,
    ) -> "RefractiveProjectionModel":
        """Construct from Phase 1 typed objects.

        Extracts raw tensors from a camera model produced by
        ``create_camera()`` and an ``InterfaceParams`` instance.

        Args:
            camera: Camera model from ``create_camera()``. Must expose ``K``,
                ``R``, and ``t`` as tensor attributes (shape (3,3), (3,3),
                (3,) respectively). Both ``_PinholeCamera`` and
                ``_FisheyeCamera`` satisfy this requirement.
            interface: Refractive interface parameters (normal, water_z,
                n_air, n_water).

        Returns:
            RefractiveProjectionModel on the same device as ``camera.K``.
        """
        return cls(
            K=camera.K,
            R=camera.R,
            t=camera.t,
            water_z=interface.water_z,
            normal=interface.normal,
            n_air=interface.n_air,
            n_water=interface.n_water,
        )

    def to(self, device: str | torch.device) -> "RefractiveProjectionModel":
        """Move all tensor attributes to the target device in-place.

        Scalar attributes (n_air, n_water, n_ratio, water_z) are Python
        floats and do not require device placement.

        Args:
            device: Target device (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns:
            self — the model mutated in-place, matching PyTorch module
            conventions.
        """
        self.K = self.K.to(device)
        self.K_inv = self.K_inv.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.C = self.C.to(device)
        self.normal = self.normal.to(device)
        return self

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project underwater 3D world points to 2D pixel coordinates.

        Uses Phase 1 Newton-Raphson (``refractive_project``) to find the
        point P on the water surface satisfying Snell's law, then projects P
        through the undistorted pinhole model.

        The Newton-Raphson loop runs for 10 fixed iterations and is fully
        differentiable (non-in-place ops throughout).

        Args:
            points: Underwater 3D points in world frame, shape (N, 3),
                float32. Z-coordinates should be greater than ``water_z``
                (below the surface in the +Z-down convention). Points above
                the surface produce invalid results (valid=False, pixels=NaN).

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
                Invalid pixels are NaN.
            valid: Boolean validity mask, shape (N,). False where the point
                is above the water surface (h_q <= 0) or the interface point
                falls behind the camera (p_cam_z <= 0).
        """
        device = points.device
        dtype = points.dtype

        interface = InterfaceParams(
            normal=self.normal,
            water_z=self.water_z,
            n_air=self.n_air,
            n_water=self.n_water,
        )

        # Find interface point P satisfying Snell's law (Newton-Raphson)
        P, _ = refractive_project(points, self.C, interface)  # (N, 3)

        # Project P through undistorted pinhole model
        p_cam = (self.R @ P.T).T + self.t.unsqueeze(0)  # (N, 3)
        p_norm = p_cam[:, :2] / p_cam[:, 2:3]  # (N, 2)
        pixels = (self.K[:2, :2] @ p_norm.T).T + self.K[:2, 2].unsqueeze(0)  # (N, 2)

        # Validity: point below water surface AND interface point in front of camera
        h_q = points[:, 2] - self.water_z  # (N,)
        valid = (h_q > 0) & (p_cam[:, 2] > 0)  # (N,)

        # Set invalid pixels to NaN
        pixels = torch.where(
            valid.unsqueeze(-1),
            pixels,
            torch.tensor(float("nan"), device=device, dtype=dtype),
        )
        return pixels, valid

    def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast refracted rays from pixel coordinates through air-water interface.

        Converts pixels to camera-frame rays via K_inv, rotates to world
        frame, intersects with the water surface at Z=water_z, then applies
        Snell's law via snells_law_3d to compute refracted directions.

        Air-to-water refraction cannot produce total internal reflection
        (n_air < n_water always), so no validity mask is returned.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points on the water surface, shape (N, 3),
                float32. Each origin has z-coordinate equal to ``water_z``.
            directions: Unit refracted ray direction vectors into water,
                shape (N, 3), float32.
        """
        n = pixels.shape[0]
        ones = torch.ones(n, 1, device=pixels.device, dtype=pixels.dtype)
        pixels_h = torch.cat([pixels, ones], dim=-1)  # (N, 3)

        # Back-project through K_inv to camera-frame rays, then normalize
        rays_cam = (self.K_inv @ pixels_h.T).T  # (N, 3)
        rays_cam = rays_cam / torch.linalg.norm(rays_cam, dim=-1, keepdim=True)

        # Rotate camera-frame rays to world frame
        rays_world = (self.R.T @ rays_cam.T).T  # (N, 3)

        # Ray-plane intersection at Z = water_z
        t_param = (self.water_z - self.C[2]) / rays_world[:, 2]  # (N,)
        origins = self.C.unsqueeze(0) + t_param.unsqueeze(-1) * rays_world  # (N, 3)

        # Snell's law: air-to-water refraction via canonical snells_law_3d
        # Air-to-water cannot produce TIR (n_air < n_water), so mask is unused
        directions, _ = snells_law_3d(rays_world, self.normal, self.n_ratio)

        return origins, directions


def project_multi(
    models: list[RefractiveProjectionModel],
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points through multiple refractive cameras.

    Loops over each camera model sequentially and stacks results. Each
    camera may be on a different device; the caller is responsible for
    ensuring ``points`` is on the correct device for each model.

    Args:
        models: List of M refractive camera models.
        points: Underwater 3D points in world frame, shape (N, 3), float32.

    Returns:
        pixels: Stacked pixel coordinates, shape (M, N, 2), float32.
            Invalid pixels are NaN.
        valid: Stacked validity masks, shape (M, N), bool.
    """
    all_pixels: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    for model in models:
        px, vl = model.project(points)
        all_pixels.append(px)
        all_valid.append(vl)
    return torch.stack(all_pixels, dim=0), torch.stack(all_valid, dim=0)


def back_project_multi(
    models: list[RefractiveProjectionModel],
    pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast refracted rays from pixels through multiple refractive cameras.

    Loops over each camera model sequentially and stacks results. Each
    camera may be on a different device; the caller is responsible for
    ensuring ``pixels`` is on the correct device for each model.

    Args:
        models: List of M refractive camera models.
        pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

    Returns:
        origins: Stacked ray origins on water surface, shape (M, N, 3),
            float32.
        directions: Stacked unit refracted directions into water,
            shape (M, N, 3), float32.
    """
    all_origins: list[torch.Tensor] = []
    all_directions: list[torch.Tensor] = []
    for model in models:
        orig, dirs = model.back_project(pixels)
        all_origins.append(orig)
        all_directions.append(dirs)
    return torch.stack(all_origins, dim=0), torch.stack(all_directions, dim=0)
