# Roadmap: AquaKit

## Overview

AquaKit is built in five phases that follow the natural dependency pyramid of the library. Phase 1 establishes every foundation type and all physics math — the dependency root that all subsequent layers require. Phase 2 builds the ProjectionModel protocol and the Newton-Raphson refractive back-projection on top of that physics. Phase 3 adds the AquaCal JSON calibration loader and undistortion, which depends on stable camera models and the projection model. Phase 4 builds the synchronized I/O layer using the calibration types from Phase 3. Phase 5 wires CI, publishing, and the rewiring guide — the release-readiness work that spans the whole codebase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation and Physics Math** - Types, camera models, transforms, Snell's law, triangulation, and known-value tests (completed 2026-02-18)
- [x] **Phase 2: Projection Protocol** - ProjectionModel protocol and batched Newton-Raphson RefractiveProjectionModel (completed 2026-02-18)
- [ ] **Phase 3: Calibration and Undistortion** - AquaCal JSON loader, CalibrationData, undistortion maps
- [ ] **Phase 4: I/O Layer** - FrameSet protocol, VideoSet, ImageSet for synchronized multi-camera access
- [ ] **Phase 5: Packaging and Release** - CI pipeline, PyPI publishing, rewiring guide
- [ ] **Phase 6: Tech Debt Cleanup** - README PyTorch install note, inline Snell's law dedup, empty test directory cleanup (GAP CLOSURE)

## Phase Details

### Phase 1: Foundation and Physics Math
**Goal**: All geometry primitives are implemented, device-agnostic, and verified against known values
**Depends on**: Nothing (first phase)
**Requirements**: TYPE-01, TYPE-02, TYPE-03, TYPE-04, TYPE-05, CAM-01, CAM-02, CAM-03, CAM-04, CAM-05, CAM-06, REF-01, REF-02, REF-03, REF-04, REF-05, REF-06, REF-07, TRN-01, TRN-02, TRN-03, TRN-04, TRI-01, TRI-02, TRI-03, QA-01, QA-02
**Success Criteria** (what must be TRUE):
  1. User can import CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2, Vec3, Mat3 from aquakit and construct instances with typed fields
  2. User can create a pinhole Camera or FisheyeCamera via create_camera() and round-trip project → back-project a 3D point to within 1e-5 pixel on both CPU and CUDA
  3. User can call snells_law_3d with a ray and interface normal and get a correctly refracted ray satisfying Snell's law (verified against known sin-ratio values); total internal reflection returns a flag, not NaN
  4. User can triangulate a 3D point from batched rays and the result matches a known ground-truth position; point-to-ray distance reports correct reprojection error
  5. All geometry tests pass on both CPU and CUDA devices with parametrized device fixtures; no test uses a hardcoded .cuda() call or imports AquaCal/AquaMVS
**Plans:** 3 plans

Plans:
- [ ] 01-01-PLAN.md — Types, interface, transforms, and test device fixture
- [ ] 01-02-PLAN.md — Pinhole and fisheye camera models with create_camera() factory
- [ ] 01-03-PLAN.md — Refraction (Snell's law, ray tracing, Newton-Raphson) and triangulation

### Phase 2: Projection Protocol
**Goal**: The refractive projection protocol and Newton-Raphson back-projection are implemented, batched, and convergence-validated
**Depends on**: Phase 1
**Requirements**: PRJ-01, PRJ-02, PRJ-03, PRJ-04
**Success Criteria** (what must be TRUE):
  1. User can construct a RefractiveProjectionModel and call project() to get pixel coordinates for underwater 3D points; result matches refractive_project from Phase 1 to floating-point tolerance
  2. User can call back_project() on pixel coordinates and recover a 3D ray; the round-trip project → back_project residual f(r_p_final) is below convergence tolerance (not just |delta| < tol)
  3. RefractiveProjectionModel accepts batched inputs (N points, M cameras) and produces correctly shaped output tensors on both CPU and CUDA
  4. Any object satisfying the ProjectionModel protocol (typed.Protocol) type-checks correctly with basedpyright without importing RefractiveProjectionModel
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — ProjectionModel protocol, RefractiveProjectionModel class, and multi-camera helpers
- [ ] 02-02-PLAN.md — Comprehensive tests for protocol compliance, convergence, and batched operations

### Phase 3: Calibration and Undistortion
**Goal**: AquaCal calibration files load into typed Python objects and images can be undistorted without any AquaCal dependency
**Depends on**: Phase 2
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04
**Success Criteria** (what must be TRUE):
  1. User can call load_calibration_data("path/to/aquacal.json") and get a CalibrationData object; aquakit is importable with AquaCal uninstalled
  2. CalibrationData.cameras returns a dict of CameraData objects; each CameraData exposes typed CameraIntrinsics, CameraExtrinsics, and InterfaceParams fields
  3. User can call compute_undistortion_maps(camera_data, image_size) and get a map pair usable with cv2.remap; maps are on the correct device
  4. User can call undistort_image(image, maps) and get an undistorted image tensor matching the source image shape
**Plans:** 2 plans

Plans:
- [ ] 03-01-PLAN.md — CameraData, CalibrationData dataclasses and load_calibration_data JSON loader
- [ ] 03-02-PLAN.md — Undistortion map computation and image remapping with PyTorch tensor I/O

### Phase 4: I/O Layer
**Goal**: Synchronized multi-camera frames are readable from video files and image directories via a common protocol
**Depends on**: Phase 3
**Requirements**: IO-01, IO-02, IO-03, IO-04
**Success Criteria** (what must be TRUE):
  1. User can construct a VideoSet from a list of video file paths and iterate frames; all cameras return the same frame index simultaneously
  2. User can construct an ImageSet from a list of image directories and iterate frames; frame tensors are float32 on the correct device
  3. Both VideoSet and ImageSet satisfy the FrameSet protocol; code written against FrameSet type-checks with either concrete class without modification
  4. Frame tensors returned from VideoSet and ImageSet are independent copies (no shared memory with OpenCV buffers that could be silently overwritten)
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — FrameSet protocol and ImageSet with synchronized image directory access
- [ ] 04-02-PLAN.md — VideoSet, create_frameset factory, and public API exports

### Phase 5: Packaging and Release
**Goal**: The library installs from PyPI, CI enforces quality on every push, and consumer teams have an import migration guide
**Depends on**: Phase 4
**Requirements**: QA-03, QA-04, QA-05
**Success Criteria** (what must be TRUE):
  1. Every push triggers a GitHub Actions workflow that runs ruff lint, basedpyright typecheck, and pytest on all supported platforms and Python versions; failures block merge
  2. A version tag triggers the PyPI trusted publishing workflow and the package appears on PyPI installable via pip install aquakit
  3. The rewiring guide lists every old AquaCal/AquaMVS import alongside its replacement aquakit import; a developer can migrate a file by find-and-replace using the table
**Plans:** 3 plans

Plans:
- [ ] 05-01-PLAN.md — Bump basedpyright to standard strictness and validate CI workflows
- [ ] 05-02-PLAN.md — Configure GitHub repo settings and PyPI trusted publishing (checkpoint)
- [ ] 05-03-PLAN.md — Write the AquaCal/AquaMVS rewiring guide

### Phase 6: Tech Debt Cleanup
**Goal**: Address low-severity tech debt items flagged by the milestone audit
**Depends on**: Phase 5
**Requirements**: None (tech debt, not requirements)
**Gap Closure**: Addresses tech debt from v1-MILESTONE-AUDIT.md
**Success Criteria** (what must be TRUE):
  1. README Quick Start section mentions installing PyTorch as a prerequisite
  2. RefractiveProjectionModel.back_project calls snells_law_3d instead of inline Snell's law copy
  3. Empty tests/e2e/ and tests/integration/ directories are either populated with meaningful tests or removed
**Plans:** 1 plan

Plans:
- [ ] 06-01-PLAN.md — README PyTorch note, Snell's law dedup, empty test directory cleanup

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation and Physics Math | 3/3 | Complete | 2026-02-18 |
| 2. Projection Protocol | 2/2 | Complete | 2026-02-18 |
| 3. Calibration and Undistortion | 0/2 | In progress | - |
| 4. I/O Layer | 0/2 | Not started | - |
| 5. Packaging and Release | 0/3 | Not started | - |
| 6. Tech Debt Cleanup | 0/1 | Not started | - |
