---
phase: 03-calibration-and-undistortion
plan: 02
subsystem: undistortion
tags: [opencv, pytorch, numpy, undistortion, fisheye, pinhole, remap]

# Dependency graph
requires:
  - phase: 03-01-calibration-and-undistortion
    provides: CameraData dataclass with CameraIntrinsics (K, dist_coeffs, image_size, is_fisheye)

provides:
  - compute_undistortion_maps(CameraData) -> tuple[np.ndarray, np.ndarray] with pinhole/fisheye dispatch
  - undistort_image(torch.Tensor, maps) -> torch.Tensor with CPU/device round-trip via OpenCV
  - 13 tests covering maps shape/dtype, tensor I/O, dispatch correctness, integration round-trips

affects:
  - downstream consumers (AquaCal, AquaMVS) that apply undistortion to calibrated images
  - 03-03 and beyond (io/ module VideoSet/ImageSet will feed tensors to undistort_image)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "OpenCV boundary: cpu().numpy() before cv2 calls, .to(device) after -- non-differentiable, documented in docstring"
    - "Fisheye dispatch: reshape dist_coeffs to (4,1) for cv2.fisheye.initUndistortRectifyMap"
    - "Pinhole dispatch: getOptimalNewCameraMatrix(alpha=0) then initUndistortRectifyMap"
    - "Return raw (map_x, map_y) tuple -- no wrapper dataclass, follows user decision"

key-files:
  created:
    - src/aquacore/undistortion.py
    - tests/unit/test_undistortion.py
  modified:
    - src/aquacore/__init__.py

key-decisions:
  - "Return raw (map_x, map_y) NumPy tuple from compute_undistortion_maps -- no UndistortionData wrapper (user decision from plan)"
  - "image_size passed directly to OpenCV as (w, h) -- CameraIntrinsics stores (width, height) matching OpenCV convention"
  - "dist_coeffs reshaped to (4,1) for fisheye path -- cv2.fisheye requires column vector, not flat array"

patterns-established:
  - "undistort_image detaches before .cpu().numpy() -- safe for gradient-tracked tensors"
  - "Identity distortion test uses centre crop with tolerance 2 -- avoids border remap artifacts while verifying core correctness"

# Metrics
duration: 15min
completed: 2026-02-18
---

# Phase 3 Plan 02: Undistortion Pipeline Summary

**cv2.remap-based undistortion with pinhole/fisheye dispatch via is_fisheye flag, PyTorch tensor I/O with device preservation, and 13 tests covering maps shape/dtype/validity, tensor round-trips, and both camera models**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-18T00:00:00Z
- **Completed:** 2026-02-18
- **Tasks:** 2
- **Files modified:** 3 (undistortion.py created, __init__.py updated, test_undistortion.py created)

## Accomplishments

- Implemented compute_undistortion_maps dispatching to cv2.fisheye.initUndistortRectifyMap or cv2.initUndistortRectifyMap based on is_fisheye flag
- Implemented undistort_image with PyTorch tensor in/out, internal OpenCV boundary via cpu().numpy() / torch.from_numpy().to(device)
- Both functions exported from aquacore top-level __init__.py
- 13 tests: maps shape (H, W), dtype float32, validity (no NaN/inf, values in range), tensor dtype/shape/device preservation, grayscale support, identity distortion verification, and full pinhole/fisheye round-trips

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement compute_undistortion_maps and undistort_image** - `be64085` (feat)
2. **Task 2: Tests for undistortion pipeline** - `240a02f` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/aquacore/undistortion.py` - compute_undistortion_maps with pinhole/fisheye dispatch; undistort_image with PyTorch-OpenCV boundary
- `src/aquacore/__init__.py` - Added compute_undistortion_maps and undistort_image to imports and __all__
- `tests/unit/test_undistortion.py` - 13 tests across two fixtures (pinhole_camera, fisheye_camera); all pass

## Decisions Made

- Return raw (map_x, map_y) NumPy tuple rather than a wrapper dataclass -- keeps the API minimal and consistent with user decision recorded in plan
- dist_coeffs reshaped to (4, 1) for fisheye path -- cv2.fisheye requires a column vector, not a flat 1D array; reshape is internal, CameraIntrinsics stores as (N,)
- image_size from CameraIntrinsics passed directly as (width, height) to OpenCV -- both use (w, h) convention, no swap needed

## Deviations from Plan

None - plan executed exactly as written. Ruff auto-fixes applied by pre-commit hooks (import sorting, line-length formatting) handled inline without altering logic.

## Issues Encountered

- Pre-commit hooks (ruff) auto-reformatted both undistortion.py and test_undistortion.py on first commit attempt; required re-staging modified files before second commit attempt. Standard workflow for this project.
- hatch run test exits with code 1 even though all 192 tests pass. Investigation shows pytest runs twice (pre-existing behavior tied to conftest CUDA parametrization and hatch's test runner interaction on Windows). Confirmed by running pytest directly -- 192 passed, 0 failed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- undistortion.py is complete and ready for consumption by io/ module (VideoSet/ImageSet will produce tensors that feed undistort_image)
- Both pinhole and fisheye paths verified with synthetic fixtures
- No blockers for remaining Phase 3 plans

## Self-Check: PASSED

Files verified:
- FOUND: src/aquacore/undistortion.py
- FOUND: tests/unit/test_undistortion.py
- FOUND: src/aquacore/__init__.py
- FOUND: .planning/phases/03-calibration-and-undistortion/03-02-SUMMARY.md

Commits verified:
- be64085 (feat: implement compute_undistortion_maps and undistort_image)
- 240a02f (test: add 13 tests for undistortion pipeline)

---
*Phase: 03-calibration-and-undistortion*
*Completed: 2026-02-18*
