---
phase: 02-projection-protocol
plan: 01
subsystem: projection
tags: [pytorch, protocol, typing, snells-law, newton-raphson, refractive]

# Dependency graph
requires:
  - phase: 01-foundation-and-physics-math
    provides: refractive_project() Newton-Raphson, InterfaceParams, _BaseCamera K/R/t attributes
provides:
  - "@runtime_checkable ProjectionModel Protocol with project() and back_project()"
  - "RefractiveProjectionModel class with __init__, from_camera, to, project, back_project"
  - "project_multi() and back_project_multi() module-level multi-camera helpers"
  - "Projection public API re-exported from both aquacore.projection and aquacore top-level"
affects:
  - 02-projection-protocol (tests plan)
  - 03-calibration-loader
  - downstream consumers: AquaCal, AquaMVS, AquaPose

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "@runtime_checkable Protocol for structural subtyping without import coupling"
    - "Thin stateful wrapper delegating Newton-Raphson to Phase 1 refractive_project()"
    - "Inline Snell's law in back_project() (AquaMVS pattern, avoids function-call overhead)"
    - "Precompute K_inv and C at construction time (not per-call)"
    - "Mutable .to(device) returning self (PyTorch module convention)"
    - "project_multi/back_project_multi sequential loop stacking into (M, N, ...) tensors"

key-files:
  created:
    - src/aquacore/projection/protocol.py
    - src/aquacore/projection/refractive.py
  modified:
    - src/aquacore/projection/__init__.py
    - src/aquacore/__init__.py

key-decisions:
  - "back_project() inlines Snell's law (AquaMVS pattern) instead of delegating to refractive_back_project() — avoids intermediate allocation, matches verified AquaMVS source"
  - "back_project() returns (origins, directions) only — no valid_mask — air-to-water TIR physically impossible"
  - "from_camera() uses Any annotation for camera parameter — avoids importing private _PinholeCamera/_FisheyeCamera, documented via docstring"
  - "project() delegates to Phase 1 refractive_project() — reuses tested 40-line Newton-Raphson, no duplication"

patterns-established:
  - "Pattern: Protocol + runtime_checkable for projection interface — structural subtyping, no import coupling"
  - "Pattern: RefractiveProjectionModel as thin wrapper over Phase 1 physics primitives"
  - "Pattern: project() two-step: Newton-Raphson interface point → undistorted pinhole projection"
  - "Pattern: Invalid pixels set to NaN, parallel boolean valid mask — consistent with Phase 1 convention"

# Metrics
duration: 25min
completed: 2026-02-18
---

# Phase 2 Plan 01: Projection Protocol and RefractiveProjectionModel Summary

**@runtime_checkable ProjectionModel Protocol wrapping Phase 1 Newton-Raphson and Snell's law into RefractiveProjectionModel with project(), back_project(), from_camera factory, and multi-camera helpers**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-02-18
- **Tasks:** 2 of 2
- **Files modified:** 4

## Accomplishments

- Implemented `@runtime_checkable ProjectionModel(Protocol)` with `project()` and `back_project()` method signatures and Google-style docstrings describing shapes and semantics
- Implemented `RefractiveProjectionModel` class with constructor (precomputing K_inv, C, n_ratio), `from_camera()` factory, `to(device)` mutable in-place transfer, `project()` delegating to Phase 1 `refractive_project()`, and `back_project()` inlining Snell's law per AquaMVS pattern
- Added `project_multi()` and `back_project_multi()` module-level helpers that loop over camera models and stack into (M, N, ...) tensors
- Updated `projection/__init__.py` and `aquacore/__init__.py` with complete exports in sorted order; all four names importable from both `aquacore.projection` and `aquacore` top-level

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement ProjectionModel protocol and RefractiveProjectionModel class** - `5e9e86f` (feat)
2. **Task 2: Update package exports** - `61cb331` (feat)

## Files Created/Modified

- `src/aquacore/projection/protocol.py` - `@runtime_checkable ProjectionModel(Protocol)` with `project()` and `back_project()` method stubs and full docstrings
- `src/aquacore/projection/refractive.py` - `RefractiveProjectionModel` class with all five methods plus `project_multi` and `back_project_multi` module functions
- `src/aquacore/projection/__init__.py` - Replaced stub with proper exports: `ProjectionModel`, `RefractiveProjectionModel`, `project_multi`, `back_project_multi`
- `src/aquacore/__init__.py` - Added projection import block (sorted between interface and refraction), four new names added to `__all__`

## Decisions Made

- **Inline Snell's law in back_project():** Chose AquaMVS pattern (inline) over delegation to `refractive_back_project()`. The Phase 1 function takes world-frame rays already, not pixels — delegation would require an intermediate conversion step anyway, and inlining matches the verified AquaMVS source exactly.
- **No valid_mask from back_project():** Protocol returns `(origins, directions)` only. Air-to-water refraction cannot produce TIR (n_air < n_water), so a mask would always be all-True for physically valid inputs. Degenerate inputs (rays parallel to surface) are caller's responsibility per device-mismatch principle.
- **from_camera() uses Any annotation:** Avoids importing private `_PinholeCamera`/`_FisheyeCamera` from the same package into the projection subpackage. Docstring documents required attributes (`K`, `R`, `t`). Both concrete classes satisfy this structurally.
- **project() delegates to refractive_project():** The Phase 1 function is tested, documented, and handles epsilon guards and autograd safety. Duplication of the 40-line Newton-Raphson loop is undesirable.

## Deviations from Plan

None - plan executed exactly as written.

The only minor resolution was import ordering: the plan said to insert the projection import "after refraction imports", but ruff's isort-compatible sort requires alphabetical order (projection < refraction). Placed projection block before refraction (p < r alphabetically). This is a linting requirement, not a semantic deviation.

## Issues Encountered

- Pre-commit ruff-format hook reformatted both `protocol.py` and `refractive.py` on first commit (CRLF/LF and minor whitespace). Re-staged and committed successfully on second attempt.
- Ruff I001 (import sort) caught out-of-order projection import in `aquacore/__init__.py` on first lint run. Fixed by moving projection block alphabetically before refraction (p < r).

## Next Phase Readiness

- `ProjectionModel` and `RefractiveProjectionModel` are fully implemented and exported; ready for Phase 2 testing plan
- `from_camera(camera, interface)` factory is available for Phase 3 calibration loader integration
- All existing Phase 1 tests still pass (64 passed, 60 CUDA-skipped)

---
*Phase: 02-projection-protocol*
*Completed: 2026-02-18*
