# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** Correct, tested PyTorch implementations of refractive multi-camera geometry that all Aqua consumers share instead of duplicating.
**Current focus:** Phase 3 - Calibration and Undistortion

## Current Position

Phase: 3 of 5 (Calibration and Undistortion)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-18 — Phase 2 complete (2/2 plans, verified)

Progress: [████░░░░░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 22 min
- Total execution time: 1.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-and-physics-math | 3 | 65 min | 22 min |
| 02-projection-protocol | 2 | ~10 min | ~5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (25 min), 01-02 (30 min), 01-03 (10 min)
- Trend: stable to fast (03 was implementation-heavy but well-researched)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: PyTorch-first, no NumPy math — one implementation, no duplication
- [Init]: Standalone tests only (no AquaCal oracle) — known-value tests, no test-time coupling
- [Init]: Datasets deferred to v2 — keeps v1 focused on geometry foundation
- [Init]: Rewiring guide, not rewiring — AquaCore ships independently
- [01-01]: Pure-PyTorch Rodrigues (not cv2.Rodrigues) — device-agnostic, autograd-compatible, handles theta=0 and theta=pi edge cases
- [01-01]: dist_coeffs stored as float64 (OpenCV requirement); K as float32 (AquaMVS convention)
- [01-01]: (output, valid_mask) return pattern for functions that can fail on individual elements (no NaN, no None)
- [01-01]: conftest.py at tests/ root (not tests/unit/) — shared by all test subdirectories
- [01-02]: OpenCV boundary: always cpu().numpy() before cv2 calls, .to(device) after — documented as non-differentiable in class docstrings
- [01-02]: atan2(|cross|, dot) for round-trip angle tests — float32 acos gives ~4.88e-4 rad noise near 1.0 even for bit-identical rays; atan2 returns exact 0.0
- [01-02]: create_camera() is sole public constructor — _PinholeCamera/_FisheyeCamera prefixed _ and not re-exported
- [01-03]: snells_law_3d orients normal internally by checking sign of cos_i — callers do not pre-orient for air→water vs water→air
- [01-03]: TIR returns (zeros, False) per (output, valid_mask) pattern — consistent with AquaMVS; not None (AquaCal pattern)
- [01-03]: refractive_project returns (N, 3) interface point — caller projects via camera model to get pixel (two-step, matches AquaMVS)
- [01-03]: TRI-03 integration uses refractive_project to find Snell's-law-correct interface points — direct line-of-sight fails due to refraction bending
- [02-02]: Round-trip test uses depth = (point_z - origin_z) / direction_z for reconstruction — no triangulation, single-model, fully deterministic
- [02-02]: NR convergence validated via re-projection residual (project reconstructed point matches original pixels atol=1e-4) — avoids duplicating residual math in tests
- [02-02]: Protocol compliance tests are device-agnostic (no device fixture) — isinstance() tests Python structure, not tensor math

### Pending Todos

None.

### Blockers/Concerns

- [Phase 3]: AquaCal JSON schema field names, shape variants (t: (3,) vs (3,1)), and optional fields must be extracted from AquaCal source before Phase 3 task planning — highest-probability integration risk
- [Phase 1]: CUDA CI runner availability must be confirmed; device-mismatch and autograd pitfalls only surface reliably on CUDA
- [Phase 1]: Glass thickness parameter resolved — simplified air-to-water model chosen (no glass layer)

## Session Continuity

Last session: 2026-02-18
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-calibration-and-undistortion/03-CONTEXT.md
