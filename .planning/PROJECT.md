# AquaKit

## What This Is

Shared PyTorch foundation library for the Aqua ecosystem (AquaCal, AquaMVS, AquaPose) providing refractive multi-camera geometry, calibration loading, synchronized I/O, and a rewiring guide for consumer migration. Published on PyPI as `aquakit`.

## Core Value

Correct, tested PyTorch implementations of refractive multi-camera geometry (Snell's law, projection, triangulation, transforms) that all Aqua consumers share instead of duplicating.

## Requirements

### Validated

- ✓ Shared types (CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3) — v1.0
- ✓ Camera models (Camera, FisheyeCamera, create_camera) — v1.0
- ✓ Interface model (air-water plane, ray-plane intersection) — v1.0
- ✓ Refraction (snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project) — v1.0
- ✓ Transforms (rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose) — v1.0
- ✓ Triangulation (triangulate_rays, triangulate_point, point_to_ray_distance — batched PyTorch) — v1.0
- ✓ Projection protocol (ProjectionModel) and RefractiveProjectionModel (Newton-Raphson) — v1.0
- ✓ Calibration loader (CalibrationData, CameraData, load from AquaCal JSON) — v1.0
- ✓ Undistortion (compute_undistortion_maps, undistort_image) — v1.0
- ✓ I/O (FrameSet protocol, VideoSet, ImageSet) — v1.0
- ✓ Standalone test suite with known-value tests (no AquaCal dependency) — v1.0
- ✓ Device-parametrized tests (CPU + CUDA) — v1.0
- ✓ CI (GitHub Actions: lint, test, typecheck) — v1.0
- ✓ PyPI publishing via trusted publishing workflow — v1.0
- ✓ Rewiring guide: old-import → new-import mapping table for AquaCal and AquaMVS — v1.0

### Active

(None — next milestone requirements TBD via `/gsd:new-milestone`)

### Out of Scope

- Datasets/synthetic data module — deferred to v2 milestone (large scope, not needed for AquaPose kickoff)
- Rewiring AquaCal/AquaMVS source code — separate project, guided by the import mapping
- Cross-validation tests importing AquaCal — standalone tests only, no test-time dependency
- NumPy API wrappers — consumers handle their own conversion at boundaries
- Mobile/embedded targets — desktop Python only
- Dome port geometry — only flat-port interface; dome port is different physics

## Context

Shipped v1.0 with 5,561 LOC Python across 37 files.
Tech stack: PyTorch, OpenCV, Hatch build system, basedpyright (standard), Ruff, GitHub Actions CI.
226 tests (all passing), device-parametrized (CPU + CUDA).
Published on PyPI. Rewiring guide covers 21 AquaCal and 8 AquaMVS symbol migrations.
Ready for AquaPose development and consumer rewiring.

## Constraints

- **Python**: >=3.11 (modern syntax: `X | Y` unions, `match` statements)
- **Tensor library**: PyTorch for all math. NumPy only at serialization boundaries (JSON, OpenCV calls).
- **Dependencies**: PyTorch, OpenCV (undistortion, image I/O), kornia (optional, image ops). No heavy ML deps (LightGlue, RoMa, Open3D stay in consumers).
- **Build system**: Hatch (already configured in pyproject.toml)
- **Type checker**: basedpyright in standard mode
- **Packaging**: PyPI via trusted publishing workflow

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch-first, no NumPy math | One implementation, no duplication. Conversion cost negligible for calibration workloads. | ✓ Good — clean single-implementation codebase |
| Standalone tests only (no AquaCal oracle) | Removes test-time coupling. Known-value tests are more reliable long-term. | ✓ Good — 226 independent tests, no external deps |
| Datasets deferred to v2 | Keeps v1 focused on geometry foundation. AquaPose can start without datasets. | ✓ Good — v1 shipped fast and focused |
| Rewiring guide, not rewiring | AquaKit ships independently. Consumer changes are a separate project. | ✓ Good — guide documents 29 symbol migrations |
| Device-follows-input convention | Low-level math follows tensor device. Consumers pass device from their config. | ✓ Good — zero device bugs in 226 tests |
| Pure-PyTorch Rodrigues | Device-agnostic, autograd-compatible, handles edge cases (theta=0, theta=pi). | ✓ Good — avoids cv2.Rodrigues CPU-only limitation |
| (output, valid_mask) return pattern | No NaN propagation; callers can batch-filter invalid elements. | ✓ Good — consistent API across refraction/TIR |
| create_camera() sole public constructor | Single entry point; _PinholeCamera/_FisheyeCamera are private. | ✓ Good — clean public API surface |
| OpenCV boundary: cpu().numpy() | All cv2 calls cross CPU boundary explicitly; documented as non-differentiable. | ✓ Good — no silent device errors |
| FrameSet structural typing (no inheritance) | ImageSet/VideoSet satisfy Protocol without inheriting — Pythonic duck typing. | ✓ Good — clean separation of interface and implementation |
| basedpyright standard mode | Stricter than basic; 0 errors without source changes needed. | ✓ Good — caught issues at CI level |

---
*Last updated: 2026-02-18 after v1.0 milestone*
