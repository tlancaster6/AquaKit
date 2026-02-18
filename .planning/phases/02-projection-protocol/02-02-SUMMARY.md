---
phase: 02-projection-protocol
plan: "02"
subsystem: testing
tags: [pytorch, projection, snells-law, newton-raphson, protocol, pytest]

# Dependency graph
requires:
  - phase: 02-01
    provides: RefractiveProjectionModel, ProjectionModel protocol, project_multi, back_project_multi
provides:
  - Protocol compliance tests (positive + negative, structural subtyping via isinstance)
  - SC-1 test: project() matches Phase 1 refractive_project pinhole projection
  - SC-2 tests: round-trip project->back_project convergence within 1e-4, NR residual via re-projection
  - SC-3 tests: batched shape contracts for N points and M cameras x N points
  - Additional tests: invalid above-water, on-axis convergence, direction unit length, from_camera(), .to(device)
affects: [03-calibration-loader, downstream consumers of projection layer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "conftest device fixture parametrizes CPU+CUDA; CUDA skips cleanly without GPU"
    - "Round-trip via depth ray extension: depth=(point_z - origin_z)/direction_z"
    - "torch.testing.assert_close for all tensor comparisons (atol, rtol=0)"
    - "@pytest.fixture (no parens) per ruff PT001 rule"

key-files:
  created:
    - tests/unit/test_projection/__init__.py
    - tests/unit/test_projection/test_protocol.py
    - tests/unit/test_projection/test_refractive.py
  modified: []

key-decisions:
  - "Round-trip test reconstructs 3D point via depth = (point_z - origin_z) / direction_z, not via triangulation — direct, single-model, no external dep"
  - "NR convergence validated via re-projection residual (project reconstructed point matches original pixels atol=1e-4) — avoids re-implementing residual math in tests"
  - "Protocol compliance tests do not use device fixture — they are device-agnostic and test Python isinstance(), not tensor math"

patterns-established:
  - "Protocol tests: always include positive (real + dummy) AND negative (missing method) cases"
  - "SC cross-validation pattern: compute expected result via Phase 1 primitives, compare with model output"

# Metrics
duration: 4min
completed: 2026-02-18
---

# Phase 2 Plan 02: Projection Protocol Tests Summary

**18-test suite covering protocol compliance, round-trip convergence within 1e-4, Newton-Raphson residual validation, and batched shape contracts for RefractiveProjectionModel**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-18T~start
- **Completed:** 2026-02-18
- **Tasks:** 2 completed
- **Files modified:** 3 created

## Accomplishments
- Protocol compliance: 4 tests verifying isinstance() works for real model, structural dummy, and two negative cases
- SC-1: project() output cross-validated against Phase 1 refractive_project + manual pinhole math (atol=1e-4)
- SC-2: round-trip and NR residual tests confirm Newton-Raphson convergence to pixel reprojection error < 1e-4
- SC-3: batched shapes verified for single-model (N=5) and multi-model (M=3, N=5) via project_multi / back_project_multi
- Additional: invalid above-water (NaN + valid=False), on-axis (epsilon guard works), direction unit length, from_camera() factory, .to(device) identity and tensor placement

## Task Commits

Each task was committed atomically:

1. **Task 1: Protocol compliance tests** - `2493a8e` (test)
2. **Task 2: RefractiveProjectionModel functional and convergence tests** - `1d9986b` (test)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `tests/unit/test_projection/__init__.py` - Test package marker
- `tests/unit/test_projection/test_protocol.py` - 4 protocol compliance tests (SC-4)
- `tests/unit/test_projection/test_refractive.py` - 14 functional tests (SC-1, SC-2, SC-3 + additional)

## Decisions Made
- Round-trip test reconstructs 3D point via `depth = (point_z - origin_z) / direction_z` — direct and deterministic, no triangulation needed
- NR convergence validated via re-projection residual rather than re-implementing residual math in tests — cleaner and exercises the full pipeline
- Protocol tests are device-agnostic (no device fixture) — isinstance() is pure Python, not tensor math

## Deviations from Plan

None — plan executed exactly as written. The only deviations were ruff auto-fixes applied by pre-commit hooks (removed `@pytest.fixture()` parens, split compound assertion, reformatted long lines). These are style-only and do not affect test semantics.

## Issues Encountered
- Pre-commit hooks (ruff lint + ruff format) required two additional `git add` cycles to absorb auto-fixes. No semantic changes.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- All four phase-2 success criteria are verified by passing tests
- Phase 3 (calibration loader) can proceed; projection layer is fully tested
- CUDA parametrization is in place; CUDA tests will run automatically when GPU available

---
*Phase: 02-projection-protocol*
*Completed: 2026-02-18*

## Self-Check: PASSED

- FOUND: tests/unit/test_projection/__init__.py
- FOUND: tests/unit/test_projection/test_protocol.py
- FOUND: tests/unit/test_projection/test_refractive.py
- FOUND: .planning/phases/02-projection-protocol/02-02-SUMMARY.md
- FOUND: commit 2493a8e (test(02-02): add ProjectionModel protocol compliance tests)
- FOUND: commit 1d9986b (test(02-02): add RefractiveProjectionModel functional and convergence tests)
