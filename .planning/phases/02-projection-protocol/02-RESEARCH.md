# Phase 2: Projection Protocol - Research

**Researched:** 2026-02-18
**Domain:** PyTorch Protocol design, refractive projection model, Newton-Raphson back-projection
**Confidence:** HIGH — research drawn directly from AquaMVS and AquaCore source code on this machine

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Method naming:**
- Protocol methods: `project()` and `back_project()` — symmetric pair, consistent with Phase 1 camera models
- AquaMVS's `cast_ray()` renamed to `back_project()` — rewiring guide will document the rename
- `project()` returns `(pixels: Tensor, valid_mask: Tensor)` — simple, no convergence metadata
- `back_project()` returns `(origins: Tensor, directions: Tensor)` — origins on water surface, directions into water

**Constructor design:**
- Primary constructor takes raw tensors: `RefractiveProjectionModel(K, R, t, water_z, normal, n_air, n_water)` — matches AquaMVS flexibility
- Factory method: `RefractiveProjectionModel.from_camera(camera, interface)` — takes Phase 1 typed objects (create_camera() result + InterfaceParams)
- Precompute and cache derived values at construction: K_inv, camera center C, n_ratio
- `.to(device)` method returns model on target device, matching AquaMVS and PyTorch conventions

**Multi-camera batching:**
- Per-camera model: one RefractiveProjectionModel per camera (proven AquaMVS pattern)
- Multi-camera helpers: `project_multi(models, points)` and `back_project_multi(models, pixels)` in same module (projection/refractive.py)
- Helpers loop sequentially over cameras, stack results into (M, N, 2) / (M, N, 3) output tensors
- NOTE: Sequential loop is v1 — future optimization could vectorize by stacking camera params into a single batched Newton-Raphson pass

**Newton-Raphson specifics:**
- Fixed 10 iterations (no early exit) — deterministic for autodiff, matches AquaMVS
- Flat interface only (no Brent fallback for tilted interfaces) — decided during project init
- Invalid pixels set to NaN in output, valid_mask is boolean — matches AquaMVS convention
- Clamp r_p to [0, r_q] per Newton-Raphson iteration — matches both AquaCal and AquaMVS
- Epsilon 1e-12 to prevent division by zero in Newton-Raphson — matches AquaMVS

### Claude's Discretion

- `back_project()` valid_mask: Claude determines whether air→water back-projection can fail and whether a mask is needed (physics says no TIR for air→water, but edge cases may exist)
- Protocol implementation: `typing.Protocol` vs `ABC` — Claude picks based on codebase patterns (AquaMVS uses `@runtime_checkable Protocol`)
- Model mutability: frozen vs mutable `.to(device)` semantics — Claude decides what fits best

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 2 implements the `ProjectionModel` protocol and `RefractiveProjectionModel` class in `src/aquacore/projection/`. The implementation is a direct adaptation of the AquaMVS `projection/` subpackage with one deliberate rename: `cast_ray()` becomes `back_project()` to form a symmetric pair with `project()`. Everything else — the Newton-Raphson algorithm, validity masking, `.to(device)` semantics, `@runtime_checkable` Protocol, precomputed derived quantities — is ported directly from the verified AquaMVS source.

Phase 1 already delivered the two functions that `RefractiveProjectionModel` builds on: `refractive_project()` (Newton-Raphson, finds interface point) and `refractive_back_project()` (ray tracing, uses `trace_ray_air_to_water()`). The Phase 2 class is a thin stateful wrapper that holds camera parameters, precomputes K_inv and C, and delegates math to those Phase 1 functions. The forward `project()` calls `refractive_project()` then does a pinhole projection of the interface point. The `back_project()` does K_inv back-projection then calls `refractive_back_project()`.

The new element unique to Phase 2 is the `from_camera()` factory method and the multi-camera helpers (`project_multi`, `back_project_multi`). The factory extracts raw tensors from Phase 1's typed objects (`_PinholeCamera | _FisheyeCamera`, `InterfaceParams`). The multi-camera helpers are simple loops that stack results — no algorithmic complexity.

**Primary recommendation:** Port AquaMVS `RefractiveProjectionModel` directly, rename `cast_ray` → `back_project`, update the Protocol signature accordingly, add `from_camera()` factory, and add multi-camera helpers. Do not diverge from AquaMVS internals — the Newton-Raphson algorithm and Snell's law inlining are verified correct.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 (in pyproject.toml) | All math — K_inv, matrix multiply, Newton-Raphson, validity masking | Project requirement; all Phase 1 math is PyTorch |
| typing.Protocol | stdlib | ProjectionModel structural interface | AquaMVS uses @runtime_checkable Protocol; PEP 544 is authoritative |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | in hatch env | Test runner | All tests |
| torch.testing.assert_close | built into torch | Tensor equality with tolerance | All geometry comparisons |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `typing.Protocol` | `abc.ABC` | Protocol is structural (no import needed for compliance); ABC requires explicit inheritance. AquaMVS uses Protocol; codebase already has ABC in camera.py for internal hierarchy. For public contract, Protocol is correct. |
| Inline Newton-Raphson in class | Delegate to `refractive_project()` | Inline gives slightly less function-call overhead but creates code duplication. Delegation reuses Phase 1's tested, documented implementation. Delegation is preferred. |
| Mutable `.to(device)` | Immutable copy semantics | AquaMVS `.to()` mutates in-place and returns self. This matches PyTorch module convention. Immutable copy would require recreating K_inv etc. Mutable is correct. |

**Installation:** No new dependencies needed — all are in pyproject.toml.

---

## Architecture Patterns

### Module-to-File Mapping

Phase 2 fills these two files (stubs already exist from project scaffold):

```
src/aquacore/projection/
├── __init__.py       # Export ProjectionModel, RefractiveProjectionModel
├── protocol.py       # @runtime_checkable class ProjectionModel(Protocol)
└── refractive.py     # RefractiveProjectionModel + project_multi + back_project_multi
```

### Pattern 1: @runtime_checkable Protocol

**What:** Define `ProjectionModel` as a `typing.Protocol` with `@runtime_checkable` decorator. This enables both static type-checking (basedpyright structural subtyping) and runtime `isinstance()` checks.

**When to use:** Always for the protocol definition. Any class with `project()` and `back_project()` methods with the right signatures automatically satisfies the protocol — no import of `ProjectionModel` needed.

**Source:** `AquaMVS/src/aquamvs/projection/protocol.py` — verified direct reference.

```python
# Source: AquaMVS/src/aquamvs/projection/protocol.py (adapted — cast_ray → back_project)
from typing import Protocol, runtime_checkable
import torch

@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol for geometric projection models."""

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world points to 2D pixel coordinates.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
                Invalid pixels are NaN.
            valid: Boolean validity mask, shape (N,).
        """
        ...

    def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast rays from pixel coordinates into the scene.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origins on water surface, shape (N, 3), float32.
            directions: Unit ray directions into water, shape (N, 3), float32.
        """
        ...
```

### Pattern 2: RefractiveProjectionModel Constructor

**What:** Primary constructor takes raw tensors. Precomputes K_inv, C, and n_ratio at construction time to avoid repeated computation on each `project()` / `back_project()` call.

**Source:** `AquaMVS/src/aquamvs/projection/refractive.py __init__` — verified direct reference.

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py (adapted)
class RefractiveProjectionModel:
    def __init__(
        self,
        K: torch.Tensor,       # (3, 3) float32
        R: torch.Tensor,       # (3, 3) float32
        t: torch.Tensor,       # (3,) float32
        water_z: float,
        normal: torch.Tensor,  # (3,) float32
        n_air: float,
        n_water: float,
    ) -> None:
        self.K = K
        self.R = R
        self.t = t
        self.water_z = water_z
        self.normal = normal
        self.n_air = n_air
        self.n_water = n_water
        # Precomputed derived quantities
        self.K_inv = torch.linalg.inv(K)        # (3, 3)
        self.C = -R.T @ t                        # (3,) camera center in world frame
        self.n_ratio = n_air / n_water           # scalar float
```

### Pattern 3: from_camera() Factory Method

**What:** Classmethod that accepts Phase 1 typed objects and extracts raw tensors for the primary constructor. This allows consumers using the calibration pipeline to construct `RefractiveProjectionModel` without manually unpacking typed objects.

**When to use:** When building from Phase 1 `create_camera()` output and `InterfaceParams`.

**Key detail:** The camera objects (`_PinholeCamera | _FisheyeCamera`) store K, R, t as attributes. The factory extracts these plus `InterfaceParams.water_z`, `InterfaceParams.normal`, `InterfaceParams.n_air`, `InterfaceParams.n_water`.

```python
# Source: AquaCore Phase 1 camera.py + types.py (interfaces researched)
from typing import TYPE_CHECKING
from ..types import InterfaceParams

@classmethod
def from_camera(
    cls,
    camera: "_PinholeCamera | _FisheyeCamera",  # Phase 1 create_camera() output
    interface: InterfaceParams,
) -> "RefractiveProjectionModel":
    """Construct from Phase 1 typed objects.

    Args:
        camera: Camera model from create_camera(). Must expose K, R, t attributes.
        interface: Refractive interface parameters.

    Returns:
        RefractiveProjectionModel on the same device as camera.K.
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
```

### Pattern 4: project() Method — Two-Step via Phase 1 refractive_project()

**What:** `project()` calls Phase 1 `refractive_project()` to find the interface point P satisfying Snell's law, then projects P through the undistorted pinhole model (`K @ (R @ P + t)`). Does NOT call `cv2.projectPoints` — the interface point is already undistorted (it is geometrically exact).

**Critical insight:** The distortion-free projection of the interface point is intentional. The refractive forward model finds where on the water surface the ray from the camera satisfies Snell's law — this is an exact geometric calculation. The AquaMVS implementation confirms this with the projection step:

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py project() lines 202-213
# Project P through pinhole model: p_cam = R @ P_world + t
p_cam = (self.R @ P.T).T + self.t.unsqueeze(0)  # (N, 3)

# Perspective division + intrinsics (no distortion — P is geometric)
p_norm = p_cam[:, :2] / p_cam[:, 2:3]  # (N, 2)
pixels = (self.K[:2, :2] @ p_norm.T).T + self.K[:2, 2].unsqueeze(0)  # (N, 2)
```

**Validity:** Point is valid when `h_q > 0` (point below water surface) AND `p_cam[:, 2] > 0` (interface point in front of camera).

**Invalid pixels:** Set to `float("nan")` via `torch.where(valid.unsqueeze(-1), pixels, nan_tensor)`.

**Differentiability:** Fully differentiable. Newton-Raphson uses non-in-place ops for autograd compatibility. The projection step is pure matrix algebra.

### Pattern 5: back_project() Method — K_inv Ray + refractive_back_project()

**What:** `back_project()` converts pixels to rays in world frame (using K_inv), then calls Phase 1 `refractive_back_project()` which calls `trace_ray_air_to_water()` for Snell's law and ray-plane intersection.

**No valid_mask needed in return value:** Air-to-water refraction cannot produce TIR (n_air < n_water always). Phase 1 `trace_ray_air_to_water()` can technically return `valid=False` for rays parallel to the water surface or pointing away from it, but these are degenerate inputs (pixels outside the camera frustum or cameras pointing up). The decision is to not return a valid mask from `back_project()` — the protocol signature is `(origins, directions)` only.

**Source:** AquaMVS `cast_ray()` is the direct reference:

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py cast_ray() (renamed back_project)
def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    N = pixels.shape[0]
    ones = torch.ones(N, 1, device=pixels.device, dtype=pixels.dtype)
    pixels_h = torch.cat([pixels, ones], dim=-1)  # (N, 3)
    rays_cam = (self.K_inv @ pixels_h.T).T         # (N, 3)
    rays_cam = rays_cam / torch.linalg.norm(rays_cam, dim=-1, keepdim=True)
    rays_world = (self.R.T @ rays_cam.T).T         # (N, 3)
    # Ray-plane intersection at Z = water_z
    t_param = (self.water_z - self.C[2]) / rays_world[:, 2]  # (N,)
    origins = self.C.unsqueeze(0) + t_param.unsqueeze(-1) * rays_world  # (N, 3)
    # Snell's law: air-to-water (n_ratio = n_air / n_water)
    cos_i = -(rays_world * self.normal).sum(dim=-1)       # (N,)
    n_oriented = -self.normal.unsqueeze(0)                 # (1, 3)
    sin_t_sq = self.n_ratio**2 * (1.0 - cos_i**2)        # (N,)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))
    directions = (
        self.n_ratio * rays_world
        + (cos_t - self.n_ratio * cos_i).unsqueeze(-1) * n_oriented
    )
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    return origins, directions
```

**Note on inline vs delegation:** AquaMVS inlines Snell's law directly in `cast_ray()` rather than calling `snells_law_3d()`. This is because the AquaMVS implementation predates Phase 1 extraction of the standalone function. For AquaCore Phase 2, consider whether to inline (faster, no function call) or delegate to `refractive_back_project()` (less duplication). Recommendation: delegate to `refractive_back_project()` from Phase 1 for code reuse, since back_project is not on the performance-critical path in typical usage.

However, note that `refractive_back_project()` has a slightly different API: it takes `pixel_rays` (already in world frame) not raw pixels. So the sequence is:
1. `K_inv` back-project pixels to camera-frame rays
2. Rotate to world frame
3. Call `refractive_back_project(pixel_rays, camera_centers, interface)` — OR just inline as AquaMVS does

Both approaches are equivalent. The delegation approach reuses Phase 1 code; the inline approach copies AquaMVS verbatim.

### Pattern 6: .to(device) — Mutable In-Place

**What:** `.to(device)` moves all internal tensors to target device and returns `self`. This matches AquaMVS semantics and PyTorch module conventions.

**Source:** `AquaMVS/src/aquamvs/projection/refractive.py to()` — verified direct reference.

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py
def to(self, device: str | torch.device) -> "RefractiveProjectionModel":
    self.K = self.K.to(device)
    self.K_inv = self.K_inv.to(device)
    self.R = self.R.to(device)
    self.t = self.t.to(device)
    self.C = self.C.to(device)
    self.normal = self.normal.to(device)
    return self
```

**Note:** `n_air`, `n_water`, `n_ratio`, and `water_z` are Python scalars (float), not tensors. They do not need device placement.

### Pattern 7: Multi-Camera Helpers

**What:** Module-level functions `project_multi()` and `back_project_multi()` that loop over a list of `RefractiveProjectionModel` instances and stack results.

**Shapes:**
- `project_multi(models: list[RefractiveProjectionModel], points: Tensor(N, 3))` → `(pixels: Tensor(M, N, 2), valid: Tensor(M, N, bool))`
- `back_project_multi(models: list[RefractiveProjectionModel], pixels: Tensor(N, 2))` → `(origins: Tensor(M, N, 3), directions: Tensor(M, N, 3))`

Where M = number of cameras, N = number of points/pixels.

```python
def project_multi(
    models: list["RefractiveProjectionModel"],
    points: torch.Tensor,  # (N, 3)
) -> tuple[torch.Tensor, torch.Tensor]:  # (M, N, 2), (M, N,)
    all_pixels = []
    all_valid = []
    for model in models:
        pixels, valid = model.project(points)
        all_pixels.append(pixels)
        all_valid.append(valid)
    return torch.stack(all_pixels, dim=0), torch.stack(all_valid, dim=0)


def back_project_multi(
    models: list["RefractiveProjectionModel"],
    pixels: torch.Tensor,  # (N, 2)
) -> tuple[torch.Tensor, torch.Tensor]:  # (M, N, 3), (M, N, 3)
    all_origins = []
    all_directions = []
    for model in models:
        origins, directions = model.back_project(pixels)
        all_origins.append(origins)
        all_directions.append(directions)
    return torch.stack(all_origins, dim=0), torch.stack(all_directions, dim=0)
```

### Pattern 8: __init__.py Export

**What:** `projection/__init__.py` must export `ProjectionModel` and `RefractiveProjectionModel` (plus `project_multi`, `back_project_multi`). The top-level `aquacore/__init__.py` should also re-export these for the public API.

**Source:** AquaMVS `projection/__init__.py` — verified direct reference.

```python
# src/aquacore/projection/__init__.py
"""Projection models for camera-to-pixel and pixel-to-ray mapping."""

from .protocol import ProjectionModel
from .refractive import (
    RefractiveProjectionModel,
    back_project_multi,
    project_multi,
)

__all__ = [
    "ProjectionModel",
    "RefractiveProjectionModel",
    "back_project_multi",
    "project_multi",
]
```

### Recommended Project Structure

```
src/aquacore/projection/
├── __init__.py       # ProjectionModel, RefractiveProjectionModel, project_multi, back_project_multi
├── protocol.py       # @runtime_checkable ProjectionModel Protocol
└── refractive.py     # RefractiveProjectionModel class + module-level multi-camera helpers
```

### Anti-Patterns to Avoid

- **Calling cv2.projectPoints in project():** The interface point P is geometrically exact and does not need distortion correction. The undistorted pinhole formula `K @ (R @ P + t)` is correct. Never call OpenCV inside `project()`.
- **In-place ops in Newton-Raphson:** Use `r_p = torch.clamp(r_p, min=0.0)` and `r_p = torch.minimum(r_p, r_q)`, NOT `r_p.clamp_()`. In-place ops break autograd.
- **Not precomputing K_inv:** Computing `torch.linalg.inv(K)` on every `back_project()` call is wasteful. Precompute at construction.
- **Not precomputing C:** Camera center `C = -R.T @ t` is used in every `project()` call. Precompute at construction.
- **Returning valid_mask from back_project():** The protocol signature returns `(origins, directions)` only. Air→water TIR is physically impossible. Do not add a third return value.
- **Not updating `.to()` when adding new tensor attributes:** If a new tensor attribute is added to the class, it must also be moved in `.to()`. Not updating `.to()` causes device mismatch errors that are hard to debug.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Newton-Raphson for forward projection | Custom NR loop | Phase 1 `refractive_project()` already implements it | Tested, documented, handles epsilon guards, autograd-safe |
| Snell's law ray tracing | Inline geometry | Phase 1 `refractive_back_project()` / `trace_ray_air_to_water()` | Tested with known-value tests; handles TIR flag, orientation |
| Protocol structural typing | ABC with explicit inheritance | `typing.Protocol` + `@runtime_checkable` | PEP 544; no import coupling; basedpyright structural checking |
| K_inv computation | Manual adjugate formula | `torch.linalg.inv(K)` | Hardware-accelerated; numerically stable |

**Key insight:** The math is entirely in Phase 1. Phase 2 is a thin stateful wrapper that holds parameters and orchestrates function calls. Do not duplicate math logic.

---

## Common Pitfalls

### Pitfall 1: Naming Collision Between Protocol and Camera back-project

**What goes wrong:** Phase 1 camera models have `pixel_to_ray()` method. Phase 2 `ProjectionModel` Protocol has `back_project()`. These are different operations: `pixel_to_ray()` returns a world-frame ray (no refraction), while `back_project()` returns a refracted ray with its origin on the water surface.

**Why it happens:** Both convert pixels to rays; the names suggest interchangeability.

**How to avoid:** Keep them distinct. `pixel_to_ray()` is on the camera model (Phase 1). `back_project()` is on the projection model (Phase 2). Document the semantic difference explicitly. The camera model does NOT implement the projection protocol.

**Warning signs:** Caller passes a `_PinholeCamera` where `ProjectionModel` is expected; `isinstance(camera, ProjectionModel)` would incorrectly return True if camera happened to have `project()` and `back_project()` methods.

### Pitfall 2: Device Mismatch at Project Time

**What goes wrong:** Model constructed on CPU. Input `points` tensor on CUDA. Matrix multiply `self.R @ points.T` raises device mismatch error.

**Why it happens:** Model stores R, t, K on whatever device they were constructed on. Input tensor may be on a different device.

**How to avoid:** Document that caller must ensure all tensors are on the same device. Provide `.to(device)` for moving the model. Do NOT add silent device transfers inside `project()` — this matches the Phase 1 principle of raising on device mismatch.

**Warning signs:** `RuntimeError: Expected all tensors to be on the same device`.

### Pitfall 3: Invalid h_c (Camera Above Water Assertion)

**What goes wrong:** Newton-Raphson in `project()` computes `h_c = water_z - C[2]`. If the camera is below the water surface (C[2] > water_z in +Z-down convention), h_c is negative, and the algorithm produces garbage.

**Why it happens:** Physics assumption: camera is in air above the water surface. AquaMVS's `refractive_project()` uses `h_c.clamp(min=1e-12)` as a silent guard.

**How to avoid:** Phase 1's `refractive_project()` already applies this clamp. Since `project()` delegates to `refractive_project()`, this is handled automatically. If implementing inline, ensure the clamp is present.

**Warning signs:** Newton-Raphson converges to nonsensical r_p values; pixels far off-image for physically valid underwater points.

### Pitfall 4: from_camera() Accessing Private Camera Attributes

**What goes wrong:** Phase 1 camera classes are `_PinholeCamera` and `_FisheyeCamera` (underscore-private). Accessing `.K`, `.R`, `.t` from outside the module is accessing private attributes.

**Why it happens:** The factory method needs raw tensors from the camera object.

**How to avoid:** These attributes ARE accessed in AquaMVS calibration loading (`.K`, `.R`, `.t` are read directly from `CameraData`). For AquaCore, `from_camera()` is in the same package, and the internal classes expose K, R, t publicly on their instances (they are plain attributes, not name-mangled). Document that `from_camera()` relies on these attributes being present on the camera object.

**Alternative:** The camera instances (`_PinholeCamera`) hold `self.K`, `self.R`, `self.t` as regular attributes (verified in camera.py line 39-41). Access is fine from within the aquacore package.

### Pitfall 5: back_project() Protocol Compliance Test Needs Both Methods

**What goes wrong:** Testing protocol compliance with `isinstance(model, ProjectionModel)` requires the Protocol to be `@runtime_checkable`. Without this decorator, `isinstance()` raises `TypeError`.

**Why it happens:** `@runtime_checkable` is opt-in in Python's `typing` module.

**How to avoid:** Always use `@runtime_checkable` on `ProjectionModel`. This is confirmed by AquaMVS protocol.py line 8.

**Warning signs:** `TypeError: Protocols with non-method members don't support issubclass()`.

### Pitfall 6: NaN Propagation into Multi-Camera Helpers

**What goes wrong:** `project_multi()` stacks outputs including NaN values for invalid points. Downstream code consuming the (M, N, 2) tensor may not check the corresponding (M, N) valid mask and inadvertently uses NaN pixels.

**Why it happens:** NaN propagates through arithmetic silently.

**How to avoid:** Document clearly that `project_multi()` returns a parallel valid mask. Callers must use valid mask before processing pixels. This matches the existing AquaCore convention from Phase 1.

---

## Code Examples

Verified patterns from source code:

### Protocol Definition (AquaMVS pattern, cast_ray → back_project rename)

```python
# Adapted from: AquaMVS/src/aquamvs/projection/protocol.py
from typing import Protocol, runtime_checkable
import torch

@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol for geometric projection models.

    Any class implementing project() and back_project() with the correct
    signatures satisfies this protocol without importing it.
    """

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D world points to 2D pixel coordinates.

        Args:
            points: 3D points in world frame, shape (N, 3), float32.

        Returns:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.
                Invalid pixels are NaN.
            valid: Boolean validity mask, shape (N,).
        """
        ...

    def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast refracted rays from pixel coordinates through air-water interface.

        Args:
            pixels: 2D pixel coordinates (u, v), shape (N, 2), float32.

        Returns:
            origins: Ray origin points on water surface, shape (N, 3), float32.
            directions: Unit ray direction vectors into water, shape (N, 3), float32.
        """
        ...
```

### RefractiveProjectionModel.project() — Full AquaMVS Implementation

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py project()
def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    dtype = points.dtype
    Q = points  # (N, 3)
    C = self.C  # (3,)

    h_c = self.water_z - C[2]          # scalar: camera height above water
    h_q = Q[:, 2] - self.water_z       # (N,): point depth below water
    dx = Q[:, 0] - C[0]               # (N,)
    dy = Q[:, 1] - C[1]               # (N,)
    r_q = torch.sqrt(dx * dx + dy * dy + 1e-12)  # (N,)
    dir_x = dx / r_q                   # (N,)
    dir_y = dy / r_q                   # (N,)

    r_p = r_q * h_c / (h_c + h_q + 1e-12)  # initial guess (N,)

    for _ in range(10):
        d_air_sq = r_p * r_p + h_c * h_c
        d_air = torch.sqrt(d_air_sq)
        r_diff = r_q - r_p
        d_water_sq = r_diff * r_diff + h_q * h_q
        d_water = torch.sqrt(d_water_sq)
        sin_air = r_p / d_air
        sin_water = r_diff / d_water
        f = self.n_air * sin_air - self.n_water * sin_water
        f_prime = (self.n_air * h_c * h_c / (d_air_sq * d_air)
                   + self.n_water * h_q * h_q / (d_water_sq * d_water))
        r_p = r_p - f / (f_prime + 1e-12)  # Newton-Raphson step
        r_p = torch.clamp(r_p, min=0.0)     # non-in-place for autograd
        r_p = torch.minimum(r_p, r_q)       # non-in-place for autograd

    # Reconstruct 3D interface point P
    px = C[0] + r_p * dir_x
    py = C[1] + r_p * dir_y
    pz = torch.full_like(px, self.water_z)
    P = torch.stack([px, py, pz], dim=-1)  # (N, 3)

    # Project P through undistorted pinhole model
    p_cam = (self.R @ P.T).T + self.t.unsqueeze(0)  # (N, 3)
    p_norm = p_cam[:, :2] / p_cam[:, 2:3]           # (N, 2)
    pixels = (self.K[:2, :2] @ p_norm.T).T + self.K[:2, 2].unsqueeze(0)  # (N, 2)

    # Validity: point below water surface AND interface point in front of camera
    valid = (h_q > 0) & (p_cam[:, 2] > 0)

    # Set invalid pixels to NaN
    pixels = torch.where(
        valid.unsqueeze(-1),
        pixels,
        torch.tensor(float("nan"), device=device, dtype=dtype),
    )
    return pixels, valid
```

### RefractiveProjectionModel.back_project() — AquaMVS cast_ray() Renamed

```python
# Source: AquaMVS/src/aquamvs/projection/refractive.py cast_ray() → back_project()
def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    N = pixels.shape[0]
    ones = torch.ones(N, 1, device=pixels.device, dtype=pixels.dtype)
    pixels_h = torch.cat([pixels, ones], dim=-1)          # (N, 3)
    rays_cam = (self.K_inv @ pixels_h.T).T                # (N, 3)
    rays_cam = rays_cam / torch.linalg.norm(rays_cam, dim=-1, keepdim=True)
    rays_world = (self.R.T @ rays_cam.T).T                # (N, 3)

    # Ray-plane intersection: origin + t_param * direction at Z = water_z
    t_param = (self.water_z - self.C[2]) / rays_world[:, 2]   # (N,)
    origins = self.C.unsqueeze(0) + t_param.unsqueeze(-1) * rays_world  # (N, 3)

    # Snell's law at interface (air → water)
    cos_i = -(rays_world * self.normal).sum(dim=-1)       # (N,)
    n_oriented = -self.normal.unsqueeze(0)                 # (1, 3)
    sin_t_sq = self.n_ratio**2 * (1.0 - cos_i**2)        # (N,)
    cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, min=0.0))
    directions = (
        self.n_ratio * rays_world
        + (cos_t - self.n_ratio * cos_i).unsqueeze(-1) * n_oriented
    )
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    return origins, directions
```

### Protocol Compliance Test (AquaMVS pattern)

```python
# Adapted from: AquaMVS/tests/test_projection/test_protocol.py
class _DummyProjectionModel:
    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)

    def back_project(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = pixels.shape[0]
        return torch.zeros(n, 3), torch.zeros(n, 3)

def test_protocol_compliance_positive():
    dummy = _DummyProjectionModel()
    assert isinstance(dummy, ProjectionModel)  # runtime_checkable

def test_protocol_compliance_missing_back_project():
    class _MissingBackProject:
        def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = points.shape[0]
            return torch.zeros(n, 2), torch.ones(n, dtype=torch.bool)
    obj = _MissingBackProject()
    assert not isinstance(obj, ProjectionModel)
```

### Round-Trip Convergence Test (AquaMVS pattern)

```python
# Adapted from: AquaMVS/tests/test_projection/test_refractive.py
def test_roundtrip_project_then_back_project(model, device):
    """project() then back_project() recovers original 3D point."""
    original_point = torch.tensor([[0.2, 0.3, 1.8]], dtype=torch.float32, device=device)

    pixels, valid = model.project(original_point)
    assert valid[0].item() is True

    origins, directions = model.back_project(pixels)

    # Depth along refracted ray (Z component is most stable)
    depth = (original_point[0, 2] - origins[0, 2]) / directions[0, 2]
    reconstructed = origins + depth * directions

    torch.testing.assert_close(reconstructed, original_point, atol=1e-4, rtol=0)
```

### Convergence Residual Test (success criterion 2)

The success criteria require verifying that the round-trip residual `f(r_p_final)` is below convergence tolerance (not just `|delta| < tol`). The residual after 10 Newton-Raphson iterations:

```python
def test_newton_raphson_residual_below_tolerance(model, device):
    """Verify Newton-Raphson converges to small Snell's law residual."""
    # Underwater point
    points = torch.tensor([[0.2, 0.3, 1.8]], dtype=torch.float32, device=device)
    # Project to get pixels
    pixels, valid = model.project(points)
    # Back-project to get rays
    origins, directions = model.back_project(pixels)
    # Reconstruct 3D point at correct depth
    depth = (points[0, 2] - origins[0, 2]) / directions[0, 2]
    reconstructed = origins + depth * directions
    # Residual: reprojected pixel should match original pixel
    reprojected, valid2 = model.project(reconstructed)
    assert valid2[0].item() is True
    torch.testing.assert_close(reprojected, pixels, atol=1e-4, rtol=0)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cast_ray()` (AquaMVS) | `back_project()` (AquaCore) | Phase 2 rename | Symmetric naming with `project()`; consistent with Phase 1 `camera.pixel_to_ray()` → `RefractiveProjectionModel.back_project()` distinction |
| No multi-camera helpers (AquaMVS) | `project_multi()`, `back_project_multi()` (AquaCore) | Phase 2 addition | Callers don't need to write their own loops for multi-camera operations |
| No `from_camera()` factory (AquaMVS) | `from_camera(camera, interface)` (AquaCore) | Phase 2 addition | Integrates with Phase 1 `create_camera()` API |
| Protocol has `cast_ray()` (AquaMVS) | Protocol has `back_project()` (AquaCore) | Phase 2 rename | Must update Protocol and all implementations |

**Deprecated/outdated:**
- `cast_ray()` method name: Replaced by `back_project()` in AquaCore. AquaMVS callers that use `cast_ray()` must be rewired to `back_project()`.

---

## Open Questions

1. **back_project() delegation vs inline Snell's law**
   - What we know: AquaMVS inlines Snell's law in `cast_ray()`. Phase 1 `refractive_back_project()` has a slightly different signature (takes world-frame rays, not pixels).
   - What's unclear: Whether to inline (copy AquaMVS verbatim) or delegate to Phase 1 function (code reuse, but requires intermediate step to convert pixels → world rays first).
   - Recommendation: Inline, following AquaMVS verbatim. The Phase 1 `refractive_back_project()` function is designed for external callers, not as an internal building block for `RefractiveProjectionModel.back_project()`. Inlining avoids intermediate allocation and keeps the implementation close to the verified AquaMVS source.

2. **Whether `project()` should delegate to Phase 1 `refractive_project()` or inline Newton-Raphson**
   - What we know: Phase 1 `refractive_project()` exists and is tested. AquaMVS inlines all Newton-Raphson logic in the class. The two implementations are nearly identical.
   - What's unclear: Delegation reuses tested code but creates a dependency on Phase 1's function signature. Inlining duplicates code but is exactly AquaMVS-verified.
   - Recommendation: Delegate to Phase 1 `refractive_project()`. The function signature is stable, it's tested, and duplication of the 40-line Newton-Raphson loop is undesirable. The thin wrapper is the correct pattern.

3. **`from_camera()` parameter type annotation**
   - What we know: Phase 1 camera classes are `_PinholeCamera | _FisheyeCamera` (private). The factory must accept either.
   - What's unclear: How to annotate the `camera` parameter in `from_camera()` without importing private classes.
   - Recommendation: Use a `Protocol` or `Any` for the type annotation. Both `_PinholeCamera` and `_FisheyeCamera` expose K, R, t as plain attributes. A structural type annotation `HasKRt = Protocol` with K/R/t attributes would be cleanest, but is overkill. Use `Any` with a docstring describing requirements, or use `_PinholeCamera | _FisheyeCamera` with a note that it's an internal type union.

---

## Sources

### Primary (HIGH confidence — direct source code inspection)

- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/protocol.py` — ProjectionModel Protocol with @runtime_checkable; cast_ray method signature
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/refractive.py` — Complete RefractiveProjectionModel: __init__, to(), cast_ray(), project() with Newton-Raphson
- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/projection/__init__.py` — Export pattern for projection subpackage
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_refractive.py` — All test patterns: constructor tests, cast_ray tests, project tests, round-trip tests, differentiability tests, CUDA tests
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_protocol.py` — Protocol compliance test patterns with _DummyProjectionModel
- `C:/Users/tucke/PycharmProjects/AquaMVS/tests/test_projection/test_cross_validation.py` — Cross-validation approach: known geometry, grid tests, rotated camera tests
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/refraction.py` — Phase 1 refractive_project(), refractive_back_project() APIs (what Phase 2 builds on)
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/types.py` — InterfaceParams, CameraExtrinsics.C property
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/camera.py` — _BaseCamera attributes K, R, t; create_camera() return type
- `C:/Users/tucke/PycharmProjects/AquaCore/tests/conftest.py` — Device fixture pattern (cpu + cuda-skipif)
- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/phases/01-foundation-and-physics-math/01-VERIFICATION.md` — Phase 1 verified artifacts and test patterns

### Secondary (MEDIUM confidence)

- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/research/aquamvs-map.md` — Pre-mapped AquaMVS architecture summary; used as navigation before source inspection
- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/research/ARCHITECTURE.md` — Layer dependency graph; confirms projection as Layer 3 above physics math

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — direct port of verified AquaMVS code; no new dependencies
- Architecture: HIGH — protocol, constructor, method signatures all copied directly from AquaMVS source
- Pitfalls: HIGH — identified from actual AquaMVS source code and Phase 1 implementation
- Test patterns: HIGH — AquaMVS test suite provides exact test structure to follow

**Research date:** 2026-02-18
**Valid until:** 2026-08-18 (stable geometry domain; PyTorch typing.Protocol API is stable)
