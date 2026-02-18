# Feature Research

**Domain:** Refractive multi-camera geometry foundation library
**Researched:** 2026-02-18
**Confidence:** HIGH (core geometry/math), MEDIUM (IO patterns), HIGH (anti-features from domain knowledge)

## Feature Landscape

### Table Stakes (Users Expect These)

Features consumers (AquaCal, AquaMVS, AquaPose) assume exist. Missing these = library fails to unblock downstream work.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Snell's law 3D refraction (`snells_law_3d`) | Every refractive geometry paper starts here; all projection/back-projection builds on it | LOW | Well-defined closed form: `n1*sin(θ1) = n2*sin(θ2)` applied to 3D ray vectors |
| Ray tracing through flat air-water interface (`trace_ray_air_to_water`) | Flat port is the dominant housing type in the Aqua ecosystem; AquaCal and AquaMVS both implement this already | MEDIUM | Depends on snells_law_3d + interface representation; handles full air→glass→water chain |
| Refractive forward projection (`refractive_project`) | AquaMVS needs to project 3D world points to 2D pixels for pose estimation and MVS | HIGH | Requires iterative solve (Newton-Raphson) or approximation; most complex operation in library |
| Refractive back-projection / ray generation (`refractive_back_project`) | AquaPose needs pixel→ray for triangulation; AquaMVS needs it for matching | HIGH | Inverse of forward projection; requires Newton-Raphson convergence; depends on interface |
| Camera intrinsics model (`CameraIntrinsics`) | All projection operations require K matrix, distortion coefficients | LOW | Standard: fx, fy, cx, cy, k1-k6, p1, p2 — OpenCV convention |
| Camera extrinsics / pose (`CameraExtrinsics`) | All world↔camera transforms require R, t | LOW | OpenCV convention: `p_cam = R @ p_world + t` |
| Interface parameters (`InterfaceParams`) | The refractive interface (normal, distance, IOR) defines the refraction geometry | LOW | Flat port: normal + distance-to-camera + n_water; parameterises the physics |
| Standard camera model (`Camera`, pinhole+distortion) | Baseline projection before/after refraction; used in calibration residuals | MEDIUM | Must support OpenCV distortion model (k1-k6, p1, p2); wraps intrinsics |
| Fisheye camera model (`FisheyeCamera`) | Underwater housings often use wide-angle fisheye lenses (>180° FOV is common in AUV rigs) | MEDIUM | OpenCV fisheye model (k1-k4 equidistant); different distortion math from pinhole |
| Pose transforms: rvec↔matrix, compose, invert | Every calibration/reconstruction workflow manipulates poses; AquaCal already uses these | LOW | `rvec_to_matrix` = Rodrigues; `compose_poses`, `invert_pose` are rigid-body ops |
| Multi-view triangulation (`triangulate_rays`, `triangulate_point`) | AquaMVS and AquaPose need 3D reconstruction from 2D correspondences | MEDIUM | DLT/SVD or midpoint; must be batched for performance in pose estimation loops |
| Point-to-ray distance (`point_to_ray_distance`) | Used in triangulation quality checks and outlier rejection | LOW | Geometric utility; depends on triangulate_rays |
| Calibration data loader (`load_calibration_data`, `CalibrationData`) | AquaCal writes JSON; AquaMVS and AquaPose read it — this is the glue | LOW-MEDIUM | JSON→typed dataclasses; must handle AquaCal's existing schema exactly |
| Undistortion maps (`compute_undistortion_maps`, `undistort_image`) | Preprocessing step expected by all consumers before geometry operations | MEDIUM | Wraps OpenCV `initUndistortRectifyMap` + `remap`; bridges NumPy↔PyTorch at boundary |
| Synchronized I/O abstractions (`FrameSet`, `VideoSet`, `ImageSet`) | All Aqua consumers read multi-camera data; need consistent abstraction | MEDIUM | FrameSet is a Protocol; VideoSet wraps cv2.VideoCapture; ImageSet handles directory trees |
| Batched PyTorch operations (device-follows-input) | AquaMVS and AquaPose run on GPU; consumers expect no hidden CPU↔GPU copies | MEDIUM | All tensor math uses input.device; no hardcoded `.cuda()`; tested on CPU+CUDA |
| ProjectionModel protocol | Consumers (AquaMVS) depend on the interface, not a concrete class; enables swapping in tests | LOW | Python `typing.Protocol`; defines `project()` + `back_project()` + `ray_from_pixel()` |
| RefractiveProjectionModel (concrete) | The actual implementation AquaMVS and AquaPose use | HIGH | Implements ProjectionModel protocol; Newton-Raphson inner loop; convergence tolerance configurable |
| Type-safe Vec2 / Vec3 / Mat3 aliases | Self-documenting function signatures in a domain with many (x,y), (x,y,z) arrays | LOW | Type aliases or lightweight TypeVar constraints; no runtime cost; improves readability |

### Differentiators (Competitive Advantage)

Features that set AquaCore apart from generic camera geometry libraries (kornia, nvTorchCam, PyTorch3D). None of these exist in any open-source Python library today.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Physically correct flat-port refraction (full air→glass→water chain) | No existing Python/PyTorch library models the multi-layer refractive interface; this is AquaCore's raison d'être | HIGH | Must handle water, glass, and air IOR separately with two Snell applications; glass thickness is a parameter |
| Newton-Raphson solver integrated into projection | Generic libraries (nvTorchCam) use closed-form distortion inversion; refractive projection has no closed form — the iterative solver *is* the differentiator | HIGH | Convergence parameters (max_iter, tolerance) are tunable; differentiable w.r.t. intrinsics for future calibration gradient flows |
| AquaCal JSON calibration schema natively supported | Consumers don't need custom parsing; the loader is the exact format AquaCal produces | LOW-MEDIUM | Idiomatic schema knowledge baked in; not a generic YAML/JSON camera loader |
| Device-parametrized test suite (CPU + CUDA) | Generic libraries test on CPU only; AquaCore validates GPU correctness for every geometry primitive | MEDIUM | pytest parametrize over `["cpu", "cuda"]`; catches `.float()` vs `.double()` bugs that only appear on GPU |
| Known-value tests for refraction | Existing libraries have no tests against ground-truth Snell's law or underwater geometry oracle; AquaCore ships with analytically-derived expected values | MEDIUM | E.g., given ray, IOR, interface normal → verify refracted direction matches hand-calculated Snell's law result |
| Synchronized multi-camera I/O with frame alignment | Most camera I/O libraries handle single cameras or assume hardware sync; AquaCore's FrameSet protocol provides a consistent abstraction over VideoSet (offline) and ImageSet (directory) | MEDIUM | Key for AquaPose which needs temporally aligned frames across 4+ cameras |
| `create_camera` factory with model dispatch | Callers pass a calibration dict; the factory selects pinhole vs fisheye based on distortion model type | LOW | Eliminates boilerplate `if model == "fisheye": FisheyeCamera(...)` in every consumer |
| Rewiring guide (old-import → new-import) | Migration from AquaCal/AquaMVS is documented; consumers can adopt incrementally without a big-bang rewrite | LOW | A mapping table in docs; not a code feature — but uniquely valuable for this ecosystem transition |

### Anti-Features (Commonly Requested, Often Problematic)

Features to explicitly NOT build in v1. These represent scope creep that would delay AquaPose without providing proportionate value.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| NumPy math wrappers (duplicating PyTorch ops in NumPy) | AquaCal is NumPy-based; developers may want drop-in replacements | Maintains two implementations, doubles test burden, blurs the PyTorch-first architecture decision. The whole point of AquaCore is one canonical PyTorch implementation | Consumers convert at their own boundaries: `torch.from_numpy()` / `.numpy()`. AquaCore exposes only PyTorch |
| Online / live camera acquisition | I/O module might be expected to handle live streams (USB cams, GigE) | Introduces hardware dependencies, driver complexity, real-time threading — all outside the foundation library's scope. Turns a geometry library into a camera SDK | Use existing hardware I/O (pypylon, OpenCV VideoCapture) and wrap results in FrameSet at caller boundary |
| Calibration optimization (bundle adjustment, parameter solving) | AquaCore ships calibration types — why not calibrate too? | Calibration optimization is AquaCal's job. Duplicating it creates split ownership and conflicts. AquaCore's role is *consuming* calibration data, not producing it | Keep `load_calibration_data` as read-only. AquaCal writes, AquaCore reads |
| Dome port geometry | Dome ports are used in some underwater systems; papers cover it | Dome port geometry is substantially more complex (sphere intersection, decentering correction) and not used in the current Aqua ecosystem. Adds significant surface area for little benefit | Defer to v2 or a separate `aquacore.geometry.dome` submodule once flat port is battle-tested |
| Synthetic data generation / rendering | AquaMVS needs synthetic training data eventually | Explicitly deferred to v2 in PROJECT.md. Requires scene description, photorealistic rendering, asset management — a separate project-sized scope | Implement as `aquacore.synthetic` in v2 milestone |
| Differentiable calibration gradient flows through projection | Researcher use case: jointly optimize IOR + pose using gradient descent | The Newton-Raphson solver can be made differentiable, but it significantly complicates the implementation and is not needed by AquaCal/AquaMVS/AquaPose v1 | Design RefractiveProjectionModel with gradient path in mind (don't block it), but don't test or document it in v1 |
| Generic camera model plugin system | Extensible architecture for arbitrary projection models | Premature abstraction. AquaCore has two concrete models (pinhole, fisheye) and one specialized model (refractive). A plugin registry adds complexity before the API is stable | ProjectionModel Protocol is the extensibility point. New models are new classes implementing the Protocol — no registration system needed |
| Mobile / embedded targets | PyTorch Lite or ONNX export for edge deployment | Constraint violation: desktop Python only (PROJECT.md). TorchScript export of Newton-Raphson iterative solvers is non-trivial and would constrain implementation choices | If edge inference is needed later, implement a lookup-table approximation in a separate module |
| Dataset classes (PyTorch Dataset subclasses) | AquaPose will need torch.utils.data.Dataset wrappers | Explicitly deferred to v2 (PROJECT.md). Datasets involve split logic, augmentation, annotation formats — a separate concern from I/O primitives | FrameSet/VideoSet/ImageSet provide the primitive building blocks; consumers wrap them in Dataset in their own codebase until v2 |

## Feature Dependencies

```
[InterfaceParams]
    └──required by──> [snells_law_3d]
                          └──required by──> [trace_ray_air_to_water]
                                                └──required by──> [refractive_project]
                                                └──required by──> [refractive_back_project]
                                                        └──required by──> [RefractiveProjectionModel]

[CameraIntrinsics]
    └──required by──> [Camera]
    └──required by──> [FisheyeCamera]
                          └──select via──> [create_camera factory]

[Camera / FisheyeCamera]
    └──required by──> [refractive_project] (applies K after ray tracing)
    └──required by──> [compute_undistortion_maps]
    └──required by──> [undistort_image]

[CameraExtrinsics]
    └──required by──> [triangulate_rays]
    └──required by──> [triangulate_point]
    └──required by──> [RefractiveProjectionModel]

[CalibrationData / load_calibration_data]
    └──produces──> [CameraIntrinsics]
    └──produces──> [CameraExtrinsics]
    └──produces──> [InterfaceParams]

[triangulate_rays]
    └──required by──> [triangulate_point]
                          └──uses──> [point_to_ray_distance] (for residual)

[ProjectionModel Protocol]
    └──implemented by──> [RefractiveProjectionModel]

[FrameSet Protocol]
    └──implemented by──> [VideoSet]
    └──implemented by──> [ImageSet]

[rvec_to_matrix] ──composes with──> [compose_poses]
[compose_poses] ──uses──> [invert_pose]
```

### Dependency Notes

- **refractive_project requires Camera**: The projection applies camera distortion *after* the refractive ray trace; Camera/FisheyeCamera handle the K-matrix and distortion step.
- **CalibrationData is the entry point**: In practice, consumers call `load_calibration_data()` and receive all three typed objects (intrinsics, extrinsics, interface). The individual types exist so consumers can construct them programmatically too (for tests, for synthetic data in v2).
- **ProjectionModel Protocol decouples consumers**: AquaMVS can type-annotate `model: ProjectionModel` and work with any future implementation (e.g., dome port) without code changes.
- **undistortion depends on Camera, not on refraction**: Undistortion maps are computed from lens distortion alone (before entering water), so they depend only on CameraIntrinsics + Camera, not on InterfaceParams.
- **triangulate_rays is independent of refraction**: It operates on rays (origin + direction) regardless of how the rays were generated (refractive or pinhole back-projection). This is intentional and keeps triangulation reusable.

## MVP Definition

### Launch With (v1)

Minimum needed for AquaPose development to begin and for AquaCal/AquaMVS to migrate.

- [ ] `types.py` — CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3 — without types nothing else type-checks
- [ ] `transforms.py` — rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose — needed by calibration loader and all pose ops
- [ ] `camera.py` — Camera, FisheyeCamera, create_camera — needed by projection
- [ ] `interface.py` — Interface, ray_plane_intersection — needed by refraction
- [ ] `refraction.py` — snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project — the core differentiator
- [ ] `projection/` — ProjectionModel protocol + RefractiveProjectionModel — what AquaMVS actually calls
- [ ] `calibration.py` — CalibrationData, load_calibration_data — what every consumer calls first
- [ ] `triangulation.py` — triangulate_rays, triangulate_point, point_to_ray_distance — needed by AquaPose
- [ ] `undistortion.py` — compute_undistortion_maps, undistort_image — needed by preprocessing pipelines
- [ ] `io/` — FrameSet protocol, VideoSet, ImageSet — needed for multi-camera data loading
- [ ] Device-parametrized tests (CPU + CUDA) — correctness guarantee for GPU consumers
- [ ] Known-value tests for Snell's law and refractive projection — validates physics correctness

### Add After Validation (v1.x)

Features to add once v1 is consumed by AquaCal and AquaMVS.

- [ ] Rewiring guide (import mapping table in docs) — trigger: first consumer migration complete
- [ ] Convergence diagnostics in Newton-Raphson solver — trigger: users report non-convergence in production
- [ ] Batch triangulation benchmark — trigger: AquaPose reports triangulation as a bottleneck

### Future Consideration (v2+)

Features explicitly deferred from v1 scope.

- [ ] Synthetic data generation (`aquacore.synthetic`) — requires rendering pipeline, scene descriptions
- [ ] PyTorch Dataset wrappers — requires knowing AquaPose's annotation format
- [ ] Dome port geometry — requires validation rig for spherical refraction
- [ ] Differentiable calibration gradient flows — requires differentiable Newton-Raphson testing harness

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| types.py (shared types) | HIGH | LOW | P1 |
| transforms.py (pose ops) | HIGH | LOW | P1 |
| refraction.py (Snell's law + trace) | HIGH | MEDIUM | P1 |
| calibration.py (JSON loader) | HIGH | LOW-MEDIUM | P1 |
| camera.py (Camera, FisheyeCamera) | HIGH | MEDIUM | P1 |
| RefractiveProjectionModel (Newton-Raphson) | HIGH | HIGH | P1 |
| triangulation.py (batched DLT/SVD) | HIGH | MEDIUM | P1 |
| io/ (FrameSet/VideoSet/ImageSet) | HIGH | MEDIUM | P1 |
| undistortion.py | MEDIUM | MEDIUM | P1 |
| Known-value tests (Snell's law, projection) | HIGH | MEDIUM | P1 |
| Device-parametrized tests (CPU+CUDA) | HIGH | LOW | P1 |
| Rewiring guide (docs) | MEDIUM | LOW | P2 |
| Convergence diagnostics | MEDIUM | LOW | P2 |
| Dome port geometry | LOW | HIGH | P3 |
| NumPy wrappers | LOW | MEDIUM | Anti-feature |
| Calibration optimization | LOW | HIGH | Anti-feature |
| Synthetic data / Dataset classes | MEDIUM | HIGH | P3 (v2) |

**Priority key:**
- P1: Must have for v1 launch
- P2: Add in v1.x after first consumers validate the core
- P3: Deferred to v2 or later milestone
- Anti-feature: Do not build

## Competitor Feature Analysis

No direct open-source Python equivalent exists for this exact domain. The closest references:

| Feature | kornia | nvTorchCam | OpenPTV | AquaCore Target |
|---------|--------|------------|---------|-----------------|
| Pinhole projection (PyTorch) | Yes (PinholeCamera) | Yes (PinholeCamera) | No (C core) | Yes |
| Fisheye projection (PyTorch) | Partial (functional only, not unified model) | Yes (OpenCVFisheyeCamera) | No | Yes — OpenCV fisheye model |
| Flat refractive interface (Snell's law) | No | No | Partial (C, multimedia geometry) | Yes — full air→glass→water |
| Newton-Raphson inverse projection | Yes (diff_newton_inverse utility) | Yes (diff_newton_inverse) | No | Yes — specialized for refractive model |
| Batched GPU triangulation | Partial (epipolar only) | No | No | Yes — DLT/SVD, batched |
| Calibration JSON loader | No | No | Proprietary format | Yes — AquaCal schema |
| Synchronized multi-camera I/O | No | No | Partial (synchronization assumed) | Yes — FrameSet/VideoSet/ImageSet |
| Device-follows-input convention | Yes (inherits PyTorch) | Yes (inherits PyTorch) | N/A | Yes — enforced by convention |
| Online/live camera acquisition | No | No | Yes (hardware trigger) | Deliberately excluded |
| Calibration optimization | Via kornia.geometry.calibration | No | Yes | Deliberately excluded (AquaCal's job) |

**Key insight (MEDIUM confidence):** The refractive multi-camera geometry domain has active academic research (ICCV 2025W, ISPRS 2025) but no dominant open-source Python library. AquaCore is building in a gap. The closest ecosystem analogue is OpenPTV, which solves a related problem (3D particle tracking through water) in C with Python bindings, but uses different conventions and does not provide PyTorch tensors.

## Sources

- nvTorchCam paper and GitHub: [nvTorchCam: An Open-source Library for Camera-Agnostic Differentiable Geometric Vision](https://arxiv.org/html/2410.12074v1) | [GitHub - NVlabs/nvTorchCam](https://github.com/NVlabs/nvTorchCam) — HIGH confidence (official repo, Oct 2024)
- Refractive calibration tool: [A Calibration Tool for Refractive Underwater Vision](https://arxiv.org/html/2405.18018v1) — HIGH confidence (peer-reviewed, ICCV 2025 Workshop)
- Kornia geometry module: [kornia.geometry documentation](https://kornia.readthedocs.io/en/latest/geometry.html) — HIGH confidence (official docs)
- OpenPTV (multimedia geometry): [Introduction - OpenPTV 0.0.9 documentation](https://openptv-python.readthedocs.io/en/latest/intro.html) — MEDIUM confidence (community project, last active 2022)
- Underwater refractive stereo: [Refractive Two-View Reconstruction for Underwater 3D Vision](https://link.springer.com/article/10.1007/s11263-019-01218-9) — HIGH confidence (IJCV peer-reviewed)
- Dome port geometry: [Refractive Geometry for Underwater Domes](https://arxiv.org/abs/2108.06575) — HIGH confidence (peer-reviewed)
- Camera calibration for underwater 3D (Snell's law ray tracing): [Camera Calibration for Underwater 3D Reconstruction Based on Ray Tracing](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w28/Pedersen_Camera_Calibration_for_CVPR_2018_paper.pdf) — HIGH confidence (CVPR Workshop)
- PyTorch3D cameras: [cameras · PyTorch3D](https://pytorch3d.org/docs/cameras) — HIGH confidence (official docs)

---
*Feature research for: Refractive multi-camera geometry foundation library (AquaCore)*
*Researched: 2026-02-18*
