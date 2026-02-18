"""Camera models (pinhole and fisheye) with project/back-project operations.

Both models use OpenCV at the NumPy boundary for distortion. All project()
and pixel_to_ray() calls are NOT differentiable due to the OpenCV CPU round-trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from .types import CameraExtrinsics, CameraIntrinsics

# ---------------------------------------------------------------------------
# Base camera model
# ---------------------------------------------------------------------------


class _BaseCamera(ABC):
    """Base camera model with shared world-to-camera and NumPy boundary logic.

    project() and pixel_to_ray() are NOT differentiable due to OpenCV CPU
    round-trip. All tensor inputs are moved to CPU for OpenCV calls and
    returned on the original device.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> None:
        self._device = intrinsics.K.device
        self.K = intrinsics.K  # (3, 3) float32
        self.dist_coeffs = intrinsics.dist_coeffs  # (N,) float64
        self.image_size = intrinsics.image_size
        self.R = extrinsics.R  # (3, 3) float32
        self.t = extrinsics.t  # (3,) float32

    def _to_cam_frame(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform world-frame points to camera frame and compute validity.

        Returns:
            p_cam: Camera-frame points, shape (N, 3).
            valid: Boolean mask, shape (N,). True where z_cam > 0.
        """
        p_cam = (self.R @ points.T).T + self.t  # (N, 3)
        valid = p_cam[:, 2] > 0  # (N,)
        return p_cam, valid

    def _to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return K and dist_coeffs as float64 numpy arrays."""
        K_np = self.K.detach().cpu().numpy().astype(np.float64)
        dist_np = self.dist_coeffs.detach().cpu().numpy().astype(np.float64)
        return K_np, dist_np

    def _rays_to_world(self, norm_pts_np: np.ndarray) -> torch.Tensor:
        """Convert normalized camera-frame 2D points to world-frame unit rays.

        Args:
            norm_pts_np: Undistorted normalized points, shape (N, 2), float32.

        Returns:
            Unit direction vectors in world frame, shape (N, 3), float32.
        """
        ones = np.ones((norm_pts_np.shape[0], 1), dtype=np.float32)
        rays_cam_np = np.concatenate([norm_pts_np, ones], axis=1)  # (N, 3)
        rays_cam = torch.from_numpy(rays_cam_np).to(self._device)
        rays_cam = rays_cam / rays_cam.norm(dim=1, keepdim=True)
        rays_world = (self.R.T @ rays_cam.T).T  # (N, 3)
        return rays_world

    def project(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world-frame points to 2D pixel coordinates.

        Args:
            points: World-frame 3D points, shape (N, 3), float32.

        Returns:
            Tuple of:
                pixels: Distorted pixel coordinates, shape (N, 2), float32.
                valid: Boolean mask, shape (N,). True where z_cam > 0.
        """
        p_cam, valid = self._to_cam_frame(points)
        K_np, dist_np = self._to_numpy()
        p_cam_np = p_cam.detach().cpu().numpy().astype(np.float64)

        pixels_np = self._cv2_project(p_cam_np, K_np, dist_np)
        pixels = torch.from_numpy(pixels_np.astype(np.float32)).to(self._device)
        return pixels, valid

    def pixel_to_ray(self, pixels: torch.Tensor) -> torch.Tensor:
        """Back-project 2D pixel coordinates to 3D world-frame unit rays.

        Args:
            pixels: Pixel coordinates, shape (N, 2), float32.

        Returns:
            Unit direction vectors in world frame, shape (N, 3), float32.
        """
        K_np, dist_np = self._to_numpy()
        pixels_np = pixels.detach().cpu().numpy().astype(np.float64)

        norm_pts_np = self._cv2_undistort(pixels_np, K_np, dist_np)
        return self._rays_to_world(norm_pts_np)

    @abstractmethod
    def _cv2_project(
        self, p_cam_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        """Run OpenCV projection. Returns pixel coords, shape (N, 2)."""

    @abstractmethod
    def _cv2_undistort(
        self, pixels_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        """Run OpenCV undistortion. Returns normalized points, shape (N, 2), float32."""


# ---------------------------------------------------------------------------
# Concrete camera models
# ---------------------------------------------------------------------------


class _PinholeCamera(_BaseCamera):
    """Pinhole camera model with OpenCV radial/tangential distortion."""

    def _cv2_project(
        self, p_cam_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        pixels_np, _ = cv2.projectPoints(
            p_cam_np,
            rvec=np.zeros(3, dtype=np.float64),
            tvec=np.zeros(3, dtype=np.float64),
            cameraMatrix=K_np,
            distCoeffs=dist_np,
        )
        return pixels_np.squeeze(1)  # (N, 1, 2) -> (N, 2)

    def _cv2_undistort(
        self, pixels_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        norm_pts_np = cv2.undistortPoints(
            pixels_np.reshape(-1, 1, 2),
            cameraMatrix=K_np,
            distCoeffs=dist_np,
        )
        return norm_pts_np.squeeze(1).astype(np.float32)  # (N, 2)


class _FisheyeCamera(_BaseCamera):
    """Fisheye camera model with OpenCV equidistant (k1-k4) distortion."""

    def _cv2_project(
        self, p_cam_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        pixels_np, _ = cv2.fisheye.projectPoints(
            p_cam_np.reshape(-1, 1, 3),  # cv2.fisheye expects (N, 1, 3)
            rvec=np.zeros(3, dtype=np.float64),
            tvec=np.zeros(3, dtype=np.float64),
            K=K_np,
            D=dist_np,
        )
        return pixels_np.squeeze(1)  # (N, 1, 2) -> (N, 2)

    def _cv2_undistort(
        self, pixels_np: np.ndarray, K_np: np.ndarray, dist_np: np.ndarray
    ) -> np.ndarray:
        norm_pts_np = cv2.fisheye.undistortPoints(
            pixels_np.reshape(-1, 1, 2),
            K=K_np,
            D=dist_np,
        )
        return norm_pts_np.squeeze(1).astype(np.float32)  # (N, 2)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_camera(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> _PinholeCamera | _FisheyeCamera:
    """Create a camera model from intrinsic and extrinsic parameters.

    This is the only public construction API for camera models. Validates
    all tensor shapes and device consistency before constructing the model.

    Args:
        intrinsics: Camera intrinsic parameters including K, dist_coeffs,
            image_size, and is_fisheye flag.
        extrinsics: Camera extrinsic parameters (R, t world-to-camera).

    Returns:
        _PinholeCamera if intrinsics.is_fisheye is False.
        _FisheyeCamera if intrinsics.is_fisheye is True.

    Raises:
        ValueError: If tensor devices do not all match, or if K/R/t/dist_coeffs
            have incorrect shapes.
    """
    # --- Device consistency check ---
    k_device = intrinsics.K.device
    r_device = extrinsics.R.device
    t_device = extrinsics.t.device
    d_device = intrinsics.dist_coeffs.device

    if k_device != r_device or k_device != t_device or k_device != d_device:
        raise ValueError(
            f"All camera tensors must be on the same device. "
            f"Got K.device={k_device}, R.device={r_device}, "
            f"t.device={t_device}, dist_coeffs.device={d_device}."
        )

    # --- Shape checks ---
    if intrinsics.K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {tuple(intrinsics.K.shape)}.")
    if extrinsics.R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {tuple(extrinsics.R.shape)}.")
    if extrinsics.t.shape != (3,):
        raise ValueError(f"t must have shape (3,), got {tuple(extrinsics.t.shape)}.")
    if intrinsics.dist_coeffs.ndim != 1:
        raise ValueError(
            f"dist_coeffs must be 1D, got shape {tuple(intrinsics.dist_coeffs.shape)}."
        )

    # --- Dispatch ---
    if intrinsics.is_fisheye:
        return _FisheyeCamera(intrinsics, extrinsics)
    return _PinholeCamera(intrinsics, extrinsics)
