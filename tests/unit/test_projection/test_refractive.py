"""Functional tests for RefractiveProjectionModel (SC-1..4 and additional).

Tests cover:
- SC-1: project() output matches Phase 1 refractive_project pinhole projection
- SC-2: Round-trip project -> back_project recovers original 3D point within 1e-4
- SC-2: Newton-Raphson residual validated via re-projection
- SC-3: Batched shapes for N points (single model) and M cameras (multi-model)
- Additional: invalid above-water points, on-axis convergence, direction unit length,
  from_camera() factory, .to(device) behaviour
"""

from __future__ import annotations

import pytest
import torch

from aquacore.camera import create_camera
from aquacore.projection import (
    RefractiveProjectionModel,
    back_project_multi,
    project_multi,
)
from aquacore.refraction import refractive_project
from aquacore.types import (
    CameraExtrinsics,
    CameraIntrinsics,
    InterfaceParams,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0
WATER_Z = 1.0
N_AIR = 1.0
N_WATER = 1.333


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def interface(device: torch.device) -> InterfaceParams:
    """Standard flat air-water interface for cross-validation."""
    return InterfaceParams(
        normal=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device),
        water_z=WATER_Z,
        n_air=N_AIR,
        n_water=N_WATER,
    )


@pytest.fixture
def model(device: torch.device) -> RefractiveProjectionModel:
    """RefractiveProjectionModel with realistic camera params, moved to device."""
    K = torch.tensor(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    m = RefractiveProjectionModel(
        K=K,
        R=R,
        t=t,
        water_z=WATER_Z,
        normal=normal,
        n_air=N_AIR,
        n_water=N_WATER,
    )
    return m.to(device)


def _make_model(device: torch.device) -> RefractiveProjectionModel:
    """Helper to construct a RefractiveProjectionModel on the given device."""
    K = torch.tensor(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    m = RefractiveProjectionModel(
        K=K, R=R, t=t, water_z=WATER_Z, normal=normal, n_air=N_AIR, n_water=N_WATER
    )
    return m.to(device)


# ---------------------------------------------------------------------------
# SC-1: project() matches Phase 1 refractive_project pinhole projection
# ---------------------------------------------------------------------------


class TestProjectMatchesPhase1:
    """SC-1: project() output must match Phase 1 refractive_project + pinhole math."""

    def test_project_matches_refractive_project(
        self,
        model: RefractiveProjectionModel,
        interface: InterfaceParams,
        device: torch.device,
    ) -> None:
        """project() matches manual K @ (R @ P + t) after Phase 1 interface point."""
        point = torch.tensor([[0.2, 0.3, 1.8]], dtype=torch.float32, device=device)

        # Model projection
        pixels, valid = model.project(point)
        assert valid[0].item(), "Point should be valid"

        # Phase 1 reference: find interface point P via Newton-Raphson
        P, _ = refractive_project(point, model.C, interface)  # (1, 3)

        # Manual pinhole projection: p_cam = R @ P + t, then K * (x/z, y/z)
        p_cam = (model.R @ P.T).T + model.t.unsqueeze(0)  # (1, 3)
        p_norm = p_cam[:, :2] / p_cam[:, 2:3]  # (1, 2)
        expected = (model.K[:2, :2] @ p_norm.T).T + model.K[:2, 2].unsqueeze(
            0
        )  # (1, 2)

        torch.testing.assert_close(pixels, expected, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# SC-2: Round-trip convergence
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """SC-2: project() -> back_project() -> reconstruct must recover original point."""

    def test_roundtrip_project_back_project(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """Round-trip via depth-parameterized ray extension recovers original point."""
        point = torch.tensor([[0.15, 0.25, 2.0]], dtype=torch.float32, device=device)

        pixels, valid = model.project(point)
        assert valid[0].item(), "Point must project to valid pixels"

        origins, directions = model.back_project(pixels)

        # Reconstruct: extend ray from origin to point's depth
        # depth = (point_z - origin_z) / direction_z
        depth = (point[0, 2] - origins[0, 2]) / directions[0, 2]
        reconstructed = origins[0] + depth * directions[0]

        torch.testing.assert_close(reconstructed, point[0], atol=1e-4, rtol=0)

    def test_newton_raphson_residual_below_tolerance(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """Re-project reconstructed 3D point — reprojection residual must be < 1e-4."""
        point = torch.tensor([[0.15, 0.25, 2.0]], dtype=torch.float32, device=device)

        # Forward: project to pixels
        pixels_orig, valid = model.project(point)
        assert valid[0].item()

        # Back-project to ray
        origins, directions = model.back_project(pixels_orig)

        # Reconstruct 3D point
        depth = (point[0, 2] - origins[0, 2]) / directions[0, 2]
        reconstructed = (origins[0] + depth * directions[0]).unsqueeze(0)  # (1, 3)

        # Re-project reconstructed point
        pixels_reprojected, valid_repr = model.project(reconstructed)
        assert valid_repr[0].item()

        # Reprojected pixels must match original — this validates NR convergence
        torch.testing.assert_close(pixels_reprojected, pixels_orig, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# SC-3: Batched shape tests
# ---------------------------------------------------------------------------


class TestBatchedShapes:
    """SC-3: project and back_project must return correct shapes for N>1 inputs."""

    def test_project_batched_shapes(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """N=5 points produce pixels (5, 2) and valid (5,)."""
        points = torch.tensor(
            [
                [0.1, 0.1, 1.5],
                [0.2, 0.0, 2.0],
                [-0.1, 0.3, 1.8],
                [0.0, 0.0, 1.2],
                [0.3, -0.2, 2.5],
            ],
            dtype=torch.float32,
            device=device,
        )
        pixels, valid = model.project(points)
        assert pixels.shape == (5, 2), f"Expected (5, 2), got {tuple(pixels.shape)}"
        assert valid.shape == (5,), f"Expected (5,), got {tuple(valid.shape)}"

    def test_back_project_batched_shapes(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """N=5 pixels produce origins (5, 3) and directions (5, 3)."""
        pixels = torch.tensor(
            [
                [320.0, 240.0],
                [350.0, 260.0],
                [290.0, 220.0],
                [400.0, 300.0],
                [250.0, 180.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        origins, directions = model.back_project(pixels)
        assert origins.shape == (5, 3), f"Expected (5, 3), got {tuple(origins.shape)}"
        assert directions.shape == (5, 3), (
            f"Expected (5, 3), got {tuple(directions.shape)}"
        )

    def test_project_multi_shapes(
        self,
        device: torch.device,
    ) -> None:
        """M=3 models x N=5 points produce pixels (3, 5, 2) and valid (3, 5)."""
        models = [_make_model(device) for _ in range(3)]
        points = torch.tensor(
            [
                [0.1, 0.1, 1.5],
                [0.2, 0.0, 2.0],
                [-0.1, 0.3, 1.8],
                [0.0, 0.0, 1.2],
                [0.3, -0.2, 2.5],
            ],
            dtype=torch.float32,
            device=device,
        )
        pixels, valid = project_multi(models, points)
        assert pixels.shape == (3, 5, 2), (
            f"Expected (3, 5, 2), got {tuple(pixels.shape)}"
        )
        assert valid.shape == (3, 5), f"Expected (3, 5), got {tuple(valid.shape)}"

    def test_back_project_multi_shapes(
        self,
        device: torch.device,
    ) -> None:
        """M=3 models x N=5 pixels produce origins (3, 5, 3) and directions (3, 5, 3)."""
        models = [_make_model(device) for _ in range(3)]
        pixels = torch.tensor(
            [
                [320.0, 240.0],
                [350.0, 260.0],
                [290.0, 220.0],
                [400.0, 300.0],
                [250.0, 180.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        origins, directions = back_project_multi(models, pixels)
        assert origins.shape == (3, 5, 3), (
            f"Expected (3, 5, 3), got {tuple(origins.shape)}"
        )
        assert directions.shape == (3, 5, 3), (
            f"Expected (3, 5, 3), got {tuple(directions.shape)}"
        )


# ---------------------------------------------------------------------------
# Additional functional tests
# ---------------------------------------------------------------------------


class TestProjectFunctional:
    """Additional project() edge-case and correctness tests."""

    def test_project_invalid_above_water(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """Point above water surface (z < water_z) produces valid=False and NaN pixels."""
        # water_z = 1.0; z=0.5 is above the surface
        point = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device=device)
        pixels, valid = model.project(point)
        assert not valid[0].item(), "Point above water should be invalid"
        assert pixels[0].isnan().all(), "Invalid pixel must be NaN"

    def test_project_on_axis_point(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """Point directly below camera (dx=dy=0) should converge; valid=True."""
        # Camera at origin (R=eye, t=zeros → C=[0,0,0]);
        # point directly below at depth 2.0 > water_z=1.0
        point = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device)
        pixels, valid = model.project(point)
        assert valid[0].item(), "On-axis point should converge and be valid"
        assert not pixels[0].isnan().any(), "On-axis pixels must not be NaN"


class TestBackProjectFunctional:
    """Additional back_project() tests."""

    def test_back_project_directions_unit_length(
        self,
        model: RefractiveProjectionModel,
        device: torch.device,
    ) -> None:
        """All direction vectors returned by back_project must have unit norm."""
        pixels = torch.tensor(
            [[320.0, 240.0], [400.0, 200.0], [250.0, 300.0]],
            dtype=torch.float32,
            device=device,
        )
        _origins, directions = model.back_project(pixels)
        norms = torch.linalg.norm(directions, dim=-1)
        ones = torch.ones(norms.shape, dtype=torch.float32, device=device)
        torch.testing.assert_close(norms, ones, atol=1e-5, rtol=0)


class TestFactoryAndDevice:
    """Tests for from_camera() factory and .to(device) behaviour."""

    def test_from_camera_matches_raw_constructor(
        self,
        device: torch.device,
    ) -> None:
        """from_camera() must produce identical project() output to raw constructor."""
        K = torch.tensor(
            [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        R = torch.eye(3, dtype=torch.float32, device=device)
        t = torch.zeros(3, dtype=torch.float32, device=device)
        normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)

        # Raw constructor
        raw_model = RefractiveProjectionModel(
            K=K,
            R=R,
            t=t,
            water_z=WATER_Z,
            normal=normal,
            n_air=N_AIR,
            n_water=N_WATER,
        )

        # from_camera factory
        intrinsics = CameraIntrinsics(
            K=K,
            dist_coeffs=torch.zeros(5, dtype=torch.float64, device=device),
            image_size=(640, 480),
            is_fisheye=False,
        )
        extrinsics = CameraExtrinsics(R=R, t=t)
        interface = InterfaceParams(
            normal=normal,
            water_z=WATER_Z,
            n_air=N_AIR,
            n_water=N_WATER,
        )
        camera = create_camera(intrinsics, extrinsics)
        factory_model = RefractiveProjectionModel.from_camera(camera, interface)

        point = torch.tensor([[0.2, 0.3, 1.8]], dtype=torch.float32, device=device)

        pixels_raw, valid_raw = raw_model.project(point)
        pixels_factory, valid_factory = factory_model.project(point)

        assert valid_raw[0].item()
        assert valid_factory[0].item()
        torch.testing.assert_close(pixels_raw, pixels_factory, atol=1e-6, rtol=0)

    def test_to_device_returns_self(
        self,
        model: RefractiveProjectionModel,
    ) -> None:
        """model.to('cpu') must return the same object (in-place, mutable semantics)."""
        result = model.to("cpu")
        assert result is model, ".to(device) must return self"

    def test_to_device_moves_tensors(
        self,
        model: RefractiveProjectionModel,
    ) -> None:
        """After .to('cpu'), all tensor attributes must be on CPU."""
        model.to("cpu")
        cpu = torch.device("cpu")
        assert model.K.device == cpu, f"K.device={model.K.device}"
        assert model.K_inv.device == cpu, f"K_inv.device={model.K_inv.device}"
        assert model.R.device == cpu, f"R.device={model.R.device}"
        assert model.t.device == cpu, f"t.device={model.t.device}"
        assert model.C.device == cpu, f"C.device={model.C.device}"
        assert model.normal.device == cpu, f"normal.device={model.normal.device}"
