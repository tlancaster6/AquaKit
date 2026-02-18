# Architecture Research

**Domain:** Refractive multi-camera geometry foundation library (PyTorch)
**Researched:** 2026-02-18
**Confidence:** MEDIUM — component structure inferred from analogous libraries (nvTorchCam, Kornia, refractive calibration tools) and domain papers; no single authoritative reference for this exact combination.

## Standard Architecture

### System Overview

The canonical structure for a PyTorch-first geometry foundation library is a strict layered pyramid. Lower layers have no knowledge of higher layers. Cross-layer communication always flows downward.

```
┌─────────────────────────────────────────────────────────────┐
│                        I/O Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  ImageSet   │  │  VideoSet    │  │  FrameSet Protocol │  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  │
│         └────────────────┴──────────────────────┘            │
├─────────────────────────────────────────────────────────────┤
│                    Calibration / Config Layer                 │
│  ┌─────────────────────┐  ┌──────────────────────────────┐   │
│  │   CalibrationData   │  │   Undistortion (remap ops)   │   │
│  │   (loader/schema)   │  │   compute_undistortion_maps  │   │
│  └──────────┬──────────┘  └──────────────┬───────────────┘   │
│             └──────────────────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                    Projection Layer                           │
│  ┌──────────────────────┐  ┌───────────────────────────────┐ │
│  │  ProjectionModel     │  │  RefractiveProjectionModel    │ │
│  │  (Protocol)          │  │  (Newton-Raphson back-proj)   │ │
│  └──────────┬───────────┘  └──────────────┬────────────────┘ │
│             └──────────────────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                    Physics / Math Layer                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌────────┐  │
│  │refraction│  │triangulat-│  │  transforms  │  │camera  │  │
│  │ (Snell)  │  │   ion     │  │ (pose/rot)   │  │(models)│  │
│  └──────┬───┘  └─────┬─────┘  └──────┬───────┘  └───┬────┘  │
│         └────────────┴───────────────┴───────────────┘       │
├─────────────────────────────────────────────────────────────┤
│                    Foundation Layer                           │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  types.py  (CameraIntrinsics, CameraExtrinsics,      │    │
│  │             InterfaceParams, Vec2, Vec3, Mat3)        │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  interface.py  (air-water plane, ray-plane intersect) │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| `types.py` | Shared named-tuple / dataclass types for all geometry primitives. No math. | Nothing — pure data structures |
| `interface.py` | Air-water plane model: plane equation, ray-plane intersection math | `types.py` |
| `camera.py` | Camera intrinsics + distortion model, `create_camera` factory | `types.py` |
| `transforms.py` | Rotation vector ↔ matrix, pose composition, pose inversion | `types.py` |
| `refraction.py` | Snell's law vector form, ray tracing through air-water interface, `refractive_project`, `refractive_back_project` | `types.py`, `interface.py`, `camera.py` |
| `triangulation.py` | Batched DLT ray intersection, point-to-ray distance | `types.py`, `transforms.py` |
| `projection/protocol.py` | `ProjectionModel` Protocol (structural subtype, not ABC) | `types.py` |
| `projection/refractive.py` | `RefractiveProjectionModel`: Newton-Raphson back-projection, wraps refraction + camera | `refraction.py`, `camera.py`, `interface.py`, `projection/protocol.py` |
| `calibration.py` | `CalibrationData`, `CameraData` dataclasses; JSON loader from AquaCal format | `types.py`, `camera.py`, `transforms.py` |
| `undistortion.py` | `compute_undistortion_maps`, `undistort_image` (wraps OpenCV remap) | `calibration.py`, `camera.py` |
| `io/frameset.py` | `FrameSet` Protocol: synchronized multi-camera frame access | `types.py` |
| `io/video.py` | `VideoSet`: FrameSet implementation over video files | `io/frameset.py`, `calibration.py` |
| `io/images.py` | `ImageSet`: FrameSet implementation over image directories | `io/frameset.py`, `calibration.py` |

## Recommended Project Structure

```
src/aquacore/
├── types.py              # Foundation: all shared types, no math
├── interface.py          # Air-water plane geometry (depends: types)
├── camera.py             # Camera models (depends: types)
├── transforms.py         # Rotation/pose utilities (depends: types)
├── refraction.py         # Snell's law + ray tracing (depends: types, interface, camera)
├── triangulation.py      # Ray intersection (depends: types, transforms)
├── calibration.py        # CalibrationData + loader (depends: types, camera, transforms)
├── undistortion.py       # Remap operations (depends: calibration, camera)
├── projection/
│   ├── __init__.py
│   ├── protocol.py       # ProjectionModel Protocol (depends: types)
│   └── refractive.py     # RefractiveProjectionModel (depends: refraction, camera, interface, protocol)
└── io/
    ├── __init__.py
    ├── frameset.py       # FrameSet Protocol (depends: types)
    ├── video.py          # VideoSet (depends: frameset, calibration)
    └── images.py         # ImageSet (depends: frameset, calibration)
```

### Structure Rationale

- **`types.py` at foundation:** Every other module imports types; making it a single file with zero dependencies avoids circular imports entirely. (HIGH confidence — universal pattern in geometry libs like Kornia, PyTorch3D)
- **`interface.py` separate from `refraction.py`:** The air-water plane model is geometrically distinct from Snell's law ray bending. Separating them allows consumers to use just the plane model without the full refraction stack.
- **`projection/` subdirectory:** nvTorchCam's `cameras_functional` pattern shows the value of separating the Protocol (contract) from the implementation. The subdirectory creates a clear boundary and allows additional projection models later.
- **`io/` subdirectory:** I/O concerns (file format, device management for video decoding) are architecturally distinct from math. Keeping them in a subdirectory prevents math modules from growing file-format dependencies.
- **`calibration.py` in the middle:** It depends on math (camera, transforms) but is depended upon by I/O. This natural mid-layer placement reflects the data flow: calibration JSON → typed parameters → frame loading.

## Architectural Patterns

### Pattern 1: Protocol-Based Abstraction (Structural Subtyping)

**What:** Define interfaces as `typing.Protocol` classes rather than ABCs. Consumers type-annotate against the Protocol; concrete classes satisfy it without inheriting.

**When to use:** `ProjectionModel` and `FrameSet` — both need interchangeable implementations (pinhole vs. refractive, video vs. images) without forcing inheritance.

**Trade-offs:**
- Pro: Zero coupling between protocol and implementation. Consumer code is testable with simple stub objects.
- Pro: Matches basedpyright's structural type checking model cleanly.
- Con: No `super()` call mechanism — shared behavior must go in a mixin or free function.

**Example:**
```python
from typing import Protocol
import torch

class ProjectionModel(Protocol):
    def project(self, points_world: torch.Tensor) -> torch.Tensor: ...
    def back_project(self, pixels: torch.Tensor) -> torch.Tensor: ...
```

### Pattern 2: Functional Math + Thin Stateful Wrapper

**What:** Implement all math as pure functions (no `self`, no side effects, no state). Wrap in a class only to hold parameters and provide the Protocol interface.

**When to use:** `refraction.py`, `triangulation.py`, `transforms.py` — all pure math. `RefractiveProjectionModel` wraps those functions.

**Trade-offs:**
- Pro: Pure functions are independently testable with known values. No fixture needed.
- Pro: Matches nvTorchCam's `cameras_functional` module approach (MEDIUM confidence from nvTorchCam architecture review).
- Con: Slight redundancy — the same logic appears as a free function and a method. Acceptable because the method is just a thin dispatch.

**Example:**
```python
# Pure function in refraction.py
def snells_law_3d(incident: torch.Tensor, normal: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
    ...

# Thin wrapper in projection/refractive.py
class RefractiveProjectionModel:
    def __init__(self, camera: Camera, interface: InterfaceParams): ...
    def project(self, points_world: torch.Tensor) -> torch.Tensor:
        return refractive_project(points_world, self.camera, self.interface)
```

### Pattern 3: Device-Agnostic Tensor Flow

**What:** All math functions infer device from input tensors. No `device` parameter. No `.cuda()` calls. Calibration loading produces CPU tensors; caller moves to device as needed.

**When to use:** Every math function in every module.

**Trade-offs:**
- Pro: Eliminates device mismatches. Caller owns device placement (matches PyTorch idiom).
- Pro: Functions compose freely — no need to propagate device through call chains.
- Con: Mixing-device inputs cause errors at the torch op level. Consumer is responsible for consistency.

**Example:**
```python
# Correct: device follows input
def triangulate_rays(origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    # No device param. Operations inherit device from origins/directions.
    ...
```

### Pattern 4: Newton-Raphson Iterative Back-Projection

**What:** Refractive back-projection has no closed-form inverse. Use Newton-Raphson (or Gauss-Newton) iteration to invert the forward projection numerically.

**When to use:** `RefractiveProjectionModel.back_project()` only. All other back-projections are analytic.

**Trade-offs:**
- Pro: Handles arbitrary refractive configurations without approximation.
- Pro: nvTorchCam uses `diff_newton_inverse` module for the same problem (MEDIUM confidence).
- Con: More expensive than analytic inversion. Must set convergence tolerances.
- Con: Iteration count and tolerance become implicit API surface — document them explicitly.

**Example structure (not full implementation):**
```python
def refractive_back_project(
    pixels: torch.Tensor,
    camera: Camera,
    interface: InterfaceParams,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> torch.Tensor:
    # Iterative Newton-Raphson solve
    ...
```

## Data Flow

### Forward Projection (3D world point → 2D pixel)

```
3D world point (torch.Tensor, shape [..., 3])
    ↓
transforms.py: apply extrinsics (R @ p + t) → camera-frame point
    ↓
interface.py: ray-plane intersection → refraction entry point
    ↓
refraction.py: snells_law_3d → refracted ray direction
    ↓
camera.py: perspective divide + distortion → image-plane point
    ↓
2D pixel (torch.Tensor, shape [..., 2])
```

### Back-Projection (2D pixel → 3D ray or point)

```
2D pixel (torch.Tensor, shape [..., 2])
    ↓
camera.py: undistort pixel → normalized camera ray
    ↓
projection/refractive.py: Newton-Raphson iterate forward model
    ↓ (converged)
refraction.py: refracted ray in world frame
    ↓
triangulation.py: intersect rays from multiple cameras → 3D point
    ↓
3D world point (torch.Tensor, shape [..., 3])
```

### Calibration Loading

```
AquaCal JSON file (disk)
    ↓  (numpy at boundary: json.load → dict → numpy arrays)
calibration.py: CalibrationData / CameraData dataclasses
    ↓  (.to_torch() or eager conversion — CPU tensors)
camera.py: Camera / FisheyeCamera instances (intrinsics as tensors)
    ↓
projection/refractive.py: RefractiveProjectionModel (consumes Camera + InterfaceParams)
```

### Frame I/O

```
VideoSet / ImageSet (file paths + CalibrationData)
    ↓  (OpenCV decode → numpy → torch.from_numpy, stays on CPU)
FrameSet.frames(index) → dict[camera_id, torch.Tensor]
    ↓
Consumer (AquaMVS, AquaPose) moves tensor to target device
```

### Key Data Flow Principles

1. **NumPy crosses boundary exactly once:** At JSON deserialization in `calibration.py` and at image decode in `io/`. All other code is pure PyTorch.
2. **Tensors travel downward:** Types → math → projection → calibration → I/O. No upward references.
3. **Device is a consumer concern:** `calibration.py` outputs CPU tensors. I/O outputs CPU tensors. Consumer calls `.to(device)` once before computation.

## Suggested Build Order

Components must be built in dependency order. Building out-of-order creates circular import risk and requires stubs.

```
Layer 0 — Foundation (no aquacore deps)
├── types.py
└── interface.py

Layer 1 — Core Math (depends on Layer 0)
├── camera.py
└── transforms.py

Layer 2 — Physics (depends on Layers 0–1)
├── refraction.py         (needs: types, interface, camera)
└── triangulation.py      (needs: types, transforms)

Layer 3 — Projection (depends on Layers 0–2)
├── projection/protocol.py    (needs: types)
└── projection/refractive.py  (needs: refraction, camera, interface, protocol)

Layer 4 — Calibration & Undistortion (depends on Layers 0–1 + optional Layer 3)
├── calibration.py            (needs: types, camera, transforms)
└── undistortion.py           (needs: calibration, camera)

Layer 5 — I/O (depends on Layers 0–4)
├── io/frameset.py            (needs: types)
├── io/video.py               (needs: frameset, calibration)
└── io/images.py              (needs: frameset, calibration)
```

**Build order implication for roadmap phases:**

- Phase 1 should deliver Layers 0–2 (types through physics math). These are pure PyTorch math; testable with known values immediately.
- Phase 2 should deliver Layer 3 (projection protocol + refractive model). Newton-Raphson convergence testing is the key gate.
- Phase 3 should deliver Layer 4 (calibration loading and undistortion). Requires real AquaCal JSON format knowledge — most likely to need schema validation work.
- Phase 4 should deliver Layer 5 (I/O). Depends on calibration shape being stable.

## Anti-Patterns

### Anti-Pattern 1: Circular Imports via Premature Abstraction

**What people do:** Put shared helper functions in a higher-level module (e.g., `calibration.py`) and then import it from a lower-level module (e.g., `camera.py`).

**Why it's wrong:** Creates circular import at package load time. Python raises `ImportError`. Common in geometry libraries that grow organically.

**Do this instead:** Shared utilities belong in `types.py` or a new `utils.py` at Layer 0. Math modules only import from lower layers.

### Anti-Pattern 2: Device-Specific Constants

**What people do:** Hardcode `torch.tensor([...], device='cuda')` or `.cuda()` inside math functions.

**Why it's wrong:** Function fails on CPU-only machines, in tests, and when consumer uses MPS. Breaks device-agnostic design.

**Do this instead:** Use `like` tensors: `torch.zeros_like(input)`, or `torch.tensor([...]).to(input.device)`. Follow device of the input argument.

### Anti-Pattern 3: NumPy Inside Math Functions

**What people do:** Use `numpy` for convenience in geometry calculations (e.g., `np.cross`, `np.linalg.solve`) inside modules that should be pure PyTorch.

**Why it's wrong:** Forces CPU evaluation; breaks autograd; adds implicit GPU→CPU→GPU round-trip cost; makes batched operations awkward.

**Do this instead:** Use `torch.linalg`, `torch.cross`, etc. NumPy is restricted to `calibration.py` (JSON boundary) and `io/` (OpenCV image decode boundary).

### Anti-Pattern 4: Monolithic Projection Module

**What people do:** Combine the projection protocol, refractive model, and pinhole fallback into one large `projection.py` file.

**Why it's wrong:** The protocol definition and each implementation have different change rates and different importers. Bundling them means any change to one implementation triggers re-testing everything.

**Do this instead:** Use the `projection/` subdirectory: `protocol.py` (stable, rarely changes), `refractive.py` (active development). Pattern is validated by nvTorchCam's `cameras` / `cameras_functional` split (MEDIUM confidence).

### Anti-Pattern 5: Implicit Interface Normal Direction

**What people do:** Hardcode `[0, 0, -1]` as the interface normal inside refraction math without documenting the coordinate convention.

**Why it's wrong:** AquaCore's coordinate system has +Z pointing down into water, so the upward normal is `[0, 0, -1]`. This is non-obvious. Undocumented constants cause sign errors when consumers work in other conventions.

**Do this instead:** Define `INTERFACE_NORMAL_DEFAULT` as a named constant in `interface.py` with an explicit docstring referencing the coordinate system.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `calibration.py` → `camera.py` | Direct call: `create_camera(intrinsics)` | Calibration constructs camera objects from loaded params |
| `io/video.py` → `calibration.py` | Direct import: `CalibrationData` type | I/O binds frames to calibrated camera IDs |
| `projection/refractive.py` → `refraction.py` | Direct function calls | Wrapper delegates math to pure functions |
| `undistortion.py` → `camera.py` | Direct call: reads distortion coefficients | Needs camera model to build remap grid |
| Consumer (AquaMVS/AquaPose) → `projection/` | Via `ProjectionModel` Protocol | Consumers accept `ProjectionModel`, not concrete class |

### External Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| AquaCal JSON → `calibration.py` | `json.load` → numpy → torch | NumPy used only at this deserialization boundary |
| OpenCV → `undistortion.py` | `cv2.remap`, `cv2.undistortPoints` | OpenCV takes numpy; convert at call site only |
| OpenCV → `io/video.py` | `cv2.VideoCapture` | Frame decode: `cv2` returns numpy → `torch.from_numpy` |
| OpenCV → `io/images.py` | `cv2.imread` | Same numpy → torch conversion at boundary |

## Scalability Considerations

This is a library, not a service. "Scaling" means handling larger batches and more cameras without API changes.

| Concern | Small batch (1-4 cameras, single frame) | Large batch (16+ cameras, video sequence) |
|---------|-----------------------------------------|-------------------------------------------|
| Memory | In-memory tensors fine | Lazy FrameSet iteration avoids loading all frames at once |
| Compute | CPU works | Device-agnostic design enables CUDA without API change |
| Parallelism | Single-process fine | Batched tensors already vectorized across cameras |

**Scaling priorities:**
1. **First bottleneck:** Newton-Raphson back-projection in loops. Mitigation: ensure `refractive_back_project` is batched over N pixels in a single call, not per-pixel.
2. **Second bottleneck:** Undistortion map computation per frame. Mitigation: `compute_undistortion_maps` is called once per camera, not per frame — document this clearly.

## Sources

- nvTorchCam architecture (MEDIUM confidence): [nvTorchCam: Camera-Agnostic Differentiable Geometric Vision](https://arxiv.org/html/2410.12074v1) — five-module design (cameras, cameras_functional, utils, warpings, diff_newton_inverse)
- Refractive calibration tool component structure (MEDIUM confidence): [A Calibration Tool for Refractive Underwater Vision](https://arxiv.org/html/2405.18018v1) — three-module pattern (camera calibration, housing calibration, stereo calibration)
- Kornia geometry module organization (HIGH confidence): [Kornia: Open Source Differentiable Computer Vision Library](https://opencv.org/blog/kornia-an-open-source-differentiable-computer-vision-library-for-pytorch/) — geometry/camera, geometry/transform, geometry/linalg separation
- Python Protocol structural subtyping (HIGH confidence): [PEP 544](https://peps.python.org/pep-0544/) — authoritative reference for Protocol-based interface design
- Build order for camera geometry (MEDIUM confidence): [Stereo Camera Calibration and Triangulation with OpenCV and Python](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html) — calibration → projection → triangulation dependency chain
- Refractive camera model self-calibration structure (LOW confidence): [NTNU Refractive Camera Model](https://ntnu-arl.github.io/refractive-camera-model/) — distortion model + virtual pinhole model separation

---
*Architecture research for: Refractive multi-camera geometry foundation library (AquaCore)*
*Researched: 2026-02-18*
