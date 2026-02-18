# Requirements: AquaCore

**Defined:** 2026-02-18
**Core Value:** Correct, tested PyTorch implementations of refractive multi-camera geometry that all Aqua consumers share instead of duplicating.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Types & Foundations

- [ ] **TYPE-01**: Library defines CameraIntrinsics dataclass with focal length, principal point, and distortion coefficients
- [ ] **TYPE-02**: Library defines CameraExtrinsics dataclass with rotation matrix and translation vector
- [ ] **TYPE-03**: Library defines InterfaceParams dataclass with interface normal, distance, and refractive indices (air, water)
- [ ] **TYPE-04**: Library defines Vec2, Vec3, Mat3 type aliases for typed tensor operations
- [ ] **TYPE-05**: Library defines coordinate system constants (world frame, camera frame, interface normal convention)

### Camera Models

- [ ] **CAM-01**: User can create a pinhole Camera that projects 3D world points to 2D pixel coordinates
- [ ] **CAM-02**: User can back-project 2D pixel coordinates to 3D rays using pinhole Camera
- [ ] **CAM-03**: Pinhole Camera handles radial and tangential distortion (OpenCV model)
- [ ] **CAM-04**: User can create a FisheyeCamera that projects 3D points using Kannala-Brandt model
- [ ] **CAM-05**: User can back-project 2D pixels to 3D rays using FisheyeCamera
- [ ] **CAM-06**: User can create appropriate camera from intrinsics via create_camera() factory

### Interface & Refraction

- [ ] **REF-01**: Library computes Snell's law in 3D (vector form) with correct normal orientation
- [ ] **REF-02**: Library handles total internal reflection gracefully (returns flag, does not NaN)
- [ ] **REF-03**: Library traces rays from air through interface into water
- [ ] **REF-04**: Library traces rays from water through interface into air
- [ ] **REF-05**: Library provides ray-plane intersection for flat air-water interface
- [ ] **REF-06**: User can refractive-project 3D underwater points to 2D pixel coordinates
- [ ] **REF-07**: User can refractive-back-project 2D pixels to 3D underwater rays

### Transforms

- [ ] **TRN-01**: User can convert rotation vectors to rotation matrices (and back)
- [ ] **TRN-02**: User can compose two poses (R1,t1) + (R2,t2) into a single pose
- [ ] **TRN-03**: User can invert a pose (R,t) → (R.T, -R.T @ t)
- [ ] **TRN-04**: User can compute camera center from extrinsics (C = -R.T @ t)

### Triangulation

- [ ] **TRI-01**: User can triangulate a 3D point from N rays (batched, N≥2)
- [ ] **TRI-02**: User can compute point-to-ray distance for reprojection error analysis
- [ ] **TRI-03**: Triangulation handles refractive rays (rays with kink at interface)

### Projection

- [ ] **PRJ-01**: Library defines ProjectionModel protocol with project() and back_project() methods
- [ ] **PRJ-02**: RefractiveProjectionModel implements forward projection through flat interface
- [ ] **PRJ-03**: RefractiveProjectionModel implements Newton-Raphson back-projection with convergence guarantees
- [ ] **PRJ-04**: RefractiveProjectionModel supports batched operations (multiple points, multiple cameras)

### Calibration

- [ ] **CAL-01**: User can load calibration data from AquaCal JSON format
- [ ] **CAL-02**: CalibrationData provides per-camera CameraData (intrinsics, extrinsics, interface params)
- [ ] **CAL-03**: User can compute undistortion maps from calibration data
- [ ] **CAL-04**: User can undistort images using precomputed maps (via OpenCV remap)

### I/O

- [ ] **IO-01**: Library defines FrameSet protocol for synchronized multi-camera frame access
- [ ] **IO-02**: User can read synchronized frames from multiple video files via VideoSet
- [ ] **IO-03**: User can read synchronized frames from multiple image directories via ImageSet
- [ ] **IO-04**: VideoSet and ImageSet implement the FrameSet protocol

### Quality & Packaging

- [ ] **QA-01**: All geometry functions have standalone tests with known input/output values
- [ ] **QA-02**: All tests parametrize over CPU and CUDA devices
- [ ] **QA-03**: CI runs lint (ruff), typecheck (basedpyright), and test (pytest) on push
- [ ] **QA-04**: Package publishes to PyPI via trusted publishing on version tag
- [ ] **QA-05**: Rewiring guide documents old-import → new-import mapping for AquaCal and AquaMVS

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Datasets & Synthetic Data

- **DAT-01**: User can generate virtual camera rigs (ring, grid, line layouts)
- **DAT-02**: User can generate object trajectories with visibility guarantees
- **DAT-03**: Library provides scene content generators (checkerboard, surfaces, skeletons)
- **DAT-04**: Library renders images through refractive interface
- **DAT-05**: Library injects configurable noise (Gaussian pixel noise, blur, exposure)
- **DAT-06**: SyntheticScenario dataclass tracks all ground truth parameters
- **DAT-07**: User can download datasets from Zenodo with caching

### Advanced Features

- **ADV-01**: Differentiable Newton-Raphson via custom autograd.Function (implicit function theorem)
- **ADV-02**: torch.compile compatibility for hot paths
- **ADV-03**: Tensor shape annotations via jaxtyping

## Out of Scope

| Feature | Reason |
|---------|--------|
| NumPy math implementations | PyTorch-only; consumers handle conversion at boundaries |
| NumPy API wrappers | Consumer responsibility, not foundation library |
| Calibration optimization/solving | Stays in AquaCal |
| Live camera acquisition | Out of scope for geometry library |
| Dome port geometry | Only flat-port interface in v1; dome port is a different physics model |
| Rewiring AquaCal/AquaMVS source | Separate project; guided by rewiring guide output |
| Cross-validation tests against AquaCal | Standalone tests only; no test-time dependency on AquaCal |
| Mobile/embedded targets | Desktop Python only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| (populated during roadmap creation) | | |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 0
- Unmapped: 33 ⚠️

---
*Requirements defined: 2026-02-18*
*Last updated: 2026-02-18 after initial definition*
