# Project Research Summary

**Project:** AquaCore
**Domain:** Refractive multi-camera geometry foundation library (Python/PyTorch)
**Researched:** 2026-02-18
**Confidence:** HIGH (stack, features, pitfalls), MEDIUM (architecture patterns)

## Executive Summary

AquaCore is a PyTorch-first geometry foundation library that solves a domain gap no existing open-source Python library addresses: physically correct multi-layer refractive projection through a flat air-glass-water interface. The library is not a standalone application but a foundation consumed by three downstream projects (AquaCal, AquaMVS, AquaPose). This constrains the design in a critical way — AquaCore must be independently installable, produce no runtime dependency on any Aqua sibling, and expose an API stable enough that downstream consumers can type-annotate against Protocol interfaces rather than concrete classes. The recommended approach is a strict five-layer pyramid (types → math → physics → projection → calibration/IO) where every layer only imports from lower layers. All geometry math is implemented as pure PyTorch functions; the Newton-Raphson solver for refractive back-projection is the highest-complexity component and must be batched, out-of-place, and device-agnostic from day one.

The stack decision is clear and well-constrained. PyTorch (>=2.6, pin 2.10) is the only viable choice for batched GPU-accelerated geometry with autograd. OpenCV headless (>=4.11) handles lens undistortion map computation. The existing Hatch/Ruff/basedpyright/pytest tooling should be kept as-is; switching to uv would provide no meaningful benefit and would break existing CI. The two optional additions that should be adopted are `jaxtyping` (shape-annotated public API) and `beartype` (dev/test runtime checking) — both kept as optional extras, not hard dependencies. NumPy is restricted to exactly two boundary crossings: AquaCal JSON deserialization in `calibration.py` and OpenCV image decode in `io/`.

The dominant risks are physics correctness bugs that produce wrong output without raising errors. Seven pitfalls are documented from direct inspection of AquaCal and AquaMVS source: normal vector sign inversion in Snell's law, camera center formula error (`-R^T@t` not `-R@t`), float32 precision loss from `torch.tensor()` vs. `torch.from_numpy()`, in-place tensor mutation breaking autograd, device mismatch for tensors created from scalar parameters, offset-camera TIR boundary stall in Newton-Raphson, and ray-depth vs. world-Z convention confusion. All seven must be addressed in the geometry foundation phase, before any higher-layer work begins. The primary mitigation is a comprehensive known-value test suite with device-parametrized fixtures (CPU + CUDA) and round-trip consistency tests (project → back_project) at multiple camera XY offsets.

## Key Findings

### Recommended Stack

The stack is mature and already partially configured in `pyproject.toml`. The only additions required are declaring `torch>=2.6` and `opencv-python-headless>=4.11` as runtime dependencies. The complete runtime surface is intentionally small — PyTorch and OpenCV headless only. Everything else (kornia, jaxtyping, beartype) is optional. This restraint is a feature: AquaCore consumers must be able to install it without pulling in ML model weights or rendering engines.

**Core technologies:**
- Python >=3.11: `X | Y` union syntax, `match` statements, `tomllib` — minimum for clean type annotations; test matrix covers 3.11, 3.12, 3.13
- PyTorch >=2.6 (pin 2.10): Batched GPU linalg, autograd, `torch.compile`; `torch.linalg.svd` handles batched triangulation natively
- OpenCV headless >=4.11 (pin 4.13.0.92): Full OpenCV distortion model, fisheye undistortion, image I/O; headless avoids Qt/X11 in server/CI environments
- Hatch/Hatchling >=1.28: Already configured; PEP 517/518 compliant; do not switch to uv_build
- Ruff 0.15.1: Already configured with correct rule set; no changes needed
- basedpyright >=1.38 (basic mode): Already configured; faster than mypy with better PyTorch stub support
- pytest >=8.3 (pin 9.0.2): Parametrize with `["cpu", "cuda"]` device strings; no pytest-pytorch plugin needed

**Optional additions (recommended as extras):**
- `jaxtyping>=0.2.35`: Shape + dtype annotations on public API (`Float[Tensor, "B 3"]`); preferred over abandoned `torchtyping`
- `beartype>=0.19`: Runtime shape checking in tests; dev-only, not a production dependency
- `kornia>=0.8.2`: Image-level ops (warp, color conversion); do NOT use its camera geometry classes

**Hard exclusions:** Open3D, PyTorch3D, LightGlue/RoMa, torchtyping, `opencv-python` (full), NumPy math inside geometry functions, SciPy for linalg, `nvTorchCam` (no refractive model).

### Expected Features

The MVP (v1) feature set is well-defined. Eleven source files/modules constitute the minimum needed for AquaPose development to begin and AquaCal/AquaMVS to migrate. Features are fully dependency-ordered in FEATURES.md.

**Must have (table stakes) — all v1:**
- `types.py`: CameraIntrinsics, CameraExtrinsics, InterfaceParams, Vec2/Vec3/Mat3 — foundation; nothing type-checks without this
- `transforms.py`: `rvec_to_matrix`, `compose_poses`, `invert_pose` — consumed by calibration loader and all pose ops
- `camera.py`: Camera (pinhole + OpenCV distortion), FisheyeCamera (Kannala-Brandt k1-k4), `create_camera` factory
- `interface.py`: Air-water plane model, ray-plane intersection
- `refraction.py`: `snells_law_3d`, `trace_ray_air_to_water`, `refractive_project`, `refractive_back_project`
- `projection/`: `ProjectionModel` Protocol + `RefractiveProjectionModel` (Newton-Raphson)
- `calibration.py`: `CalibrationData`, `load_calibration_data` — AquaCal JSON schema, standalone (no AquaCal import)
- `triangulation.py`: Batched `triangulate_rays`, `triangulate_point`, `point_to_ray_distance`
- `undistortion.py`: `compute_undistortion_maps`, `undistort_image` (wraps OpenCV remap)
- `io/`: `FrameSet` Protocol, `VideoSet`, `ImageSet` — synchronized multi-camera data loading
- Device-parametrized tests (CPU + CUDA) and known-value tests for Snell's law and refractive projection

**Should have (competitive differentiators):**
- Physically correct full air→glass→water chain (multi-layer, glass thickness as parameter) — no existing Python library does this
- Offset-camera round-trip test suite (XY offsets up to 0.5 m) — existing AquaCal bug demonstrates why this is a differentiator
- `create_camera` factory with model dispatch (eliminates consumer boilerplate)
- Rewiring guide (import mapping for AquaCal/AquaMVS migration)
- Convergence diagnostics in Newton-Raphson (add in v1.x post-validation)

**Defer (v2+):**
- Synthetic data generation (`aquacore.synthetic`) — requires rendering pipeline
- PyTorch Dataset wrappers — requires knowing AquaPose's annotation format
- Dome port geometry — not in current Aqua ecosystem; spherical refraction is substantially more complex
- Differentiable calibration gradient flows through Newton-Raphson — design to not block this, but don't test/document in v1

**Explicit anti-features (do not build):**
- NumPy math wrappers, online/live camera acquisition, calibration optimization, generic plugin system, mobile/embedded targets, calibration bundle adjustment

### Architecture Approach

The architecture is a strict five-layer downward-only dependency pyramid. The layering eliminates circular import risk and ensures every layer is independently testable. All geometry math is implemented as pure functions with no state; `RefractiveProjectionModel` is a thin stateful wrapper that delegates to those functions and satisfies the `ProjectionModel` Protocol. NumPy crosses the boundary exactly twice (JSON deserialization, OpenCV image decode) and appears nowhere else.

**Layer structure (build in this order):**
1. **Foundation** (`types.py`, `interface.py`): Shared types and air-water plane geometry; zero internal imports
2. **Core Math** (`camera.py`, `transforms.py`): Camera models and rotation/pose utilities; depend only on Layer 0
3. **Physics** (`refraction.py`, `triangulation.py`): Snell's law, ray tracing, batched DLT triangulation; depend on Layers 0-1
4. **Projection** (`projection/protocol.py`, `projection/refractive.py`): Protocol interface and Newton-Raphson implementation; depend on Layers 0-2
5. **Calibration and IO** (`calibration.py`, `undistortion.py`, `io/`): AquaCal JSON loader and multi-camera frame I/O; depend on Layers 0-3

**Key patterns:**
- `typing.Protocol` for `ProjectionModel` and `FrameSet` — structural subtyping, zero coupling between protocol and implementation
- Pure functions + thin stateful wrapper — all math is independently testable; `RefractiveProjectionModel` just dispatches
- Device-agnostic tensor flow — device inferred from input; no `device=` parameter on math functions; no `.cuda()` calls
- Newton-Raphson iterative back-projection — the only non-analytic operation; must be batched over N pixels per call, not per-pixel

**Key anti-patterns to avoid:**
- Circular imports via upward references (any module importing from a higher layer)
- Hardcoded device constants inside math functions
- NumPy inside geometry ops
- Monolithic `projection.py` (use the subdirectory split)
- Implicit interface normal direction without documented coordinate convention

### Critical Pitfalls

All seven pitfalls below are sourced from direct AquaCal/AquaMVS code inspection and documented bugs. All are silent failures — wrong output, no exception.

1. **Normal vector sign convention inversion in Snell's law** — The stored normal `[0,0,-1]` (toward air) must be flipped to `[0,0,+1]` (into water) before applying the vectorized formula. AquaCal does this via `cos_i = -(rays * normal).sum(dim=-1)`. Preserve this pattern exactly. Verify with: `sin(30°) air → sin(22°) water` known-value test.

2. **Camera center formula: `C = -R^T @ t`, not `-R @ t`** — Using `-R @ t` gives geometrically wrong output silently, and tests with `R = I` never catch it. Add non-identity rotation tests with cameras at known world positions.

3. **`torch.tensor(numpy_array)` silently downcasts to float32** — Use `torch.from_numpy(arr)` (preserves dtype) at all NumPy boundaries. Newton-Raphson converges to 1e-9 m in float64 but only 1e-6 m in float32; triangulation loses ~3 orders of magnitude in float32.

4. **In-place tensor mutation breaks autograd** — The Newton-Raphson loop must use out-of-place ops throughout (`r_p = torch.clamp(r_p, ...)` not `r_p.clamp_()`). AquaMVS already does this correctly. Validate with a `requires_grad=True` Jacobian test.

5. **Offset camera TIR boundary stall** — For cameras at XY offsets >0.2 m, the Newton-Raphson initial guess overshoots and the iteration clamps to a wrong boundary answer (no exception). Check `f(r_p_final) < tolerance`, not just `|delta| < tolerance`. Include round-trip tests at 0, 0.1, 0.3, 0.5 m offsets.

6. **Device mismatch for tensor creation from scalar parameters** — `torch.full(size, self.water_z)` without `device=points.device` creates a CPU tensor that fails on CUDA silently in CI (CPU-only runners). Audit every tensor creation in the Newton-Raphson loop; use `torch.full_like` / `torch.zeros_like`.

7. **Ray depth vs. world Z confusion** — For oblique rays, ray depth `d` differs from world Z displacement by `cos(theta)`. Name every quantity precisely in docstrings; test oblique rays explicitly.

## Implications for Roadmap

Based on research, the dependency order from ARCHITECTURE.md maps cleanly to four roadmap phases. All critical pitfalls (Pitfalls 1-7) fall into Phase 1 because they live in the math that all higher layers depend on. Getting geometry correct before building projection and IO is the key ordering constraint.

### Phase 1: Foundation and Physics Math

**Rationale:** Types, transforms, camera models, and Snell's law are the dependency root. Nothing else can be built or tested without them. All seven critical pitfalls are in this layer — fix them here before they can propagate upward. This phase is also the riskiest numerically; establishing float dtype policy, normal sign convention, and device-agnostic patterns here prevents cascading bugs.

**Delivers:** `types.py`, `interface.py`, `camera.py`, `transforms.py`, `refraction.py`, `triangulation.py` — and the full known-value + device-parametrized test suite.

**Addresses features:** CameraIntrinsics, CameraExtrinsics, InterfaceParams, Camera, FisheyeCamera, snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project (functional form), triangulate_rays, triangulate_point, point_to_ray_distance, pose transforms.

**Avoids:** Normal sign inversion (Pitfall 1), camera center formula (Pitfall 2), float32 precision loss (Pitfall 3), in-place autograd corruption (Pitfall 4), offset camera TIR stall (Pitfall 6), ray depth confusion (Pitfall 7).

**Research flag:** Standard PyTorch geometry patterns — well-documented. Snell's law vectorized form has one canonical implementation. Skip `/gsd:research-phase` for this phase; the math is deterministic.

### Phase 2: Projection Protocol and RefractiveProjectionModel

**Rationale:** The Newton-Raphson refractive back-projection is the library's core differentiator and highest implementation complexity. It depends on all of Phase 1. Building the Protocol first (stable, rarely changes) then the concrete model (active development) matches the nvTorchCam `cameras` / `cameras_functional` split pattern. The convergence correctness gate (checking `f(r_p_final) < tolerance`, not just `|delta|`) is Phase 1's test signal that Phase 2 can begin.

**Delivers:** `projection/protocol.py` (ProjectionModel Protocol), `projection/refractive.py` (RefractiveProjectionModel with batched Newton-Raphson), convergence tolerance documentation.

**Addresses features:** ProjectionModel Protocol, RefractiveProjectionModel, `create_camera` factory integration, device-parametrized projection tests.

**Avoids:** Device mismatch (Pitfall 5 — all tensor creation in Newton-Raphson loop must include `device=points.device`), in-place autograd corruption (Pitfall 4 — verify with Jacobian test against finite differences).

**Research flag:** The Newton-Raphson convergence and batching implementation has a concrete reference in AquaMVS (`_refractive_project_newton_batch`). Recommend reviewing that implementation directly before writing Phase 2 code. No additional research phase needed.

### Phase 3: Calibration Loader and Undistortion

**Rationale:** The AquaCal JSON schema is the integration contract with the Aqua ecosystem. This phase depends on stable camera models (Phase 1) and produces the typed inputs that IO (Phase 4) binds to. The loader must be standalone — `import aquacore` must work with AquaCal uninstalled. This phase also resolves the most ambiguous external dependency: the exact AquaCal JSON schema, including `t` vector shape variants `(3,)` vs. `(3, 1)` and per-camera `water_z` consistency checks.

**Delivers:** `calibration.py` (CalibrationData, load_calibration_data, AquaCal JSON schema parsed directly), `undistortion.py` (compute_undistortion_maps, undistort_image).

**Addresses features:** CalibrationData, load_calibration_data, compute_undistortion_maps, undistort_image, `create_camera` integration.

**Avoids:** Calibration loader coupling to AquaCal package (Pitfall in "Looks Done But Isn't" checklist), OpenCV `cv2.undistortPoints` normalized coordinate bug (returns normalized coords, not pixel coords — re-apply K), fisheye `D` shape `(4,1)` requirement.

**Research flag:** The AquaCal JSON schema must be read from the actual AquaCal source before implementation. This is the phase most likely to need schema validation work. Recommend a targeted schema extraction from AquaCal before planning Phase 3 tasks.

### Phase 4: IO Layer (FrameSet, VideoSet, ImageSet)

**Rationale:** IO is the outermost layer and depends on stable calibration types from Phase 3. The FrameSet Protocol enables synchronized multi-camera data loading — a differentiator no existing Python camera library provides cleanly. Building IO last ensures all inner layers are stable before adding file-format and video-decode concerns.

**Delivers:** `io/frameset.py` (FrameSet Protocol), `io/video.py` (VideoSet over video files), `io/images.py` (ImageSet over image directories).

**Addresses features:** FrameSet Protocol, VideoSet, ImageSet, synchronized multi-camera I/O, `torch.from_numpy` memory sharing safety (`.clone()` after decode).

**Avoids:** `torch.from_numpy` memory sharing corruption (must `.clone()` frames that OpenCV may modify), OpenCV image size convention `(width, height)` vs. NumPy `(height, width)` swap.

**Research flag:** FrameSet, VideoSet, and ImageSet are straightforward abstractions over `cv2.VideoCapture` and `cv2.imread`. No additional research phase needed; patterns are well-established.

### Phase Ordering Rationale

- **Physics before projection:** The Newton-Raphson solver iterates the forward projection; you must have a correct `refractive_project` (Phase 1) before implementing `refractive_back_project` as an iterative inverse (Phase 2).
- **Projection before calibration:** `calibration.py` constructs `Camera` and `InterfaceParams` objects and then hands them to `RefractiveProjectionModel`. The model's constructor must exist before the loader can be tested end-to-end.
- **Calibration before IO:** `VideoSet` and `ImageSet` bind frames to `CalibrationData` camera IDs. The data types must be stable before the IO layer can reference them.
- **All seven pitfalls in Phase 1:** Every critical pitfall lives in the geometry math that all subsequent layers depend on. Catching sign errors, dtype issues, and device mismatches at Phase 1 ensures Phases 2-4 build on a correct foundation.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (Calibration):** AquaCal JSON schema must be extracted from the actual AquaCal source before task planning. The exact field names, shapes, and optionality of `water_z`, `t`, `R`, distortion coefficients, and camera model type discriminators are not fully documented and must be confirmed.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation and Physics):** Snell's law, DLT triangulation, rotation vectors, and PyTorch device-agnostic conventions are all well-documented. The implementation reference (AquaCal, AquaMVS) is available locally.
- **Phase 2 (Projection):** AquaMVS contains a working reference implementation of the batched Newton-Raphson solver. Pattern is clear; the Protocol/implementation split is well-established (nvTorchCam precedent).
- **Phase 4 (IO):** OpenCV VideoCapture and imread abstractions are standard. The FrameSet Protocol is a straightforward Python Protocol definition.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions confirmed on PyPI as of 2026-02-18. Tooling already configured in repo. No controversial decisions. |
| Features | HIGH | Feature set derived from actual consumer requirements (AquaCal, AquaMVS, AquaPose) and validated against competitor libraries. No existing Python library covers refractive geometry. |
| Architecture | MEDIUM | Layer structure inferred from analogous libraries (nvTorchCam, Kornia) and domain papers. No single authoritative reference for this exact combination. The five-layer pyramid is a strong pattern but specific module boundaries are design decisions, not established fact. |
| Pitfalls | HIGH | All seven critical pitfalls sourced from direct AquaCal and AquaMVS source inspection and documented bug history. The offset-camera TIR bug (Pitfall 6) has a specific 2026-02-04 fix as evidence. |

**Overall confidence:** HIGH for what to build and what to avoid. MEDIUM for exact module boundaries and internal APIs, which will be refined during roadmap phase planning.

### Gaps to Address

- **AquaCal JSON schema specifics:** The exact field names, shape variants, and optional fields in the calibration JSON must be extracted from AquaCal source before Phase 3 task planning. A schema mismatch discovered during implementation is the highest-probability integration risk.
- **jaxtyping exact version:** PyPI returned an error during stack research; version was cross-checked as ~0.2.38 via WebSearch. Verify the exact current version at `https://pypi.org/project/jaxtyping/` before declaring it in `pyproject.toml`.
- **CUDA CI runner:** Seven pitfalls (device mismatch, autograd in-place corruption) are only reliably caught on CUDA. If the GitHub Actions workflow lacks a CUDA runner, these pitfalls will only surface in production. The existing `.github/workflows/test.yml` should be reviewed before Phase 1 testing is declared complete.
- **Glass thickness parameter:** The full air→glass→water chain requires a glass thickness parameter in `InterfaceParams`. Research confirms this is a differentiator but does not specify the default or valid range. This should be clarified when defining `InterfaceParams` in Phase 1.
- **`water_z` scalar vs. tensor in `RefractiveProjectionModel`:** The "Looks Done But Isn't" checklist identifies `water_z` as an ambiguous type (Python float vs. 0-dim tensor). The convention should be decided in Phase 1 and documented in `InterfaceParams`.

## Sources

### Primary (HIGH confidence)

- AquaCal source (`refractive_geometry.py`, `interface_model.py`, calibration tests) — Newton-Raphson pattern, normal sign convention, TIR bug fix
- AquaMVS source (`projection/refractive.py`, `triangulation.py`, `calibration.py`) — batched Newton-Raphson, determinant filter, dtype preservation
- PyPI: torch 2.10.0, opencv-python-headless 4.13.0.92, kornia 0.8.2, ruff 0.15.1, basedpyright 1.38.1, pytest 9.0.2, hatchling 1.28.0
- PyTorch `torch.linalg` docs — batched SVD, solve, stable API since torch 1.9
- OpenCV camera calibration docs — `p_cam = R @ p_world + t` convention, `C = -R^T @ t`
- PEP 544 — authoritative Protocol structural subtyping reference
- jaxtyping recommendation (Patrick Kidger, author) — deprecation of torchtyping confirmed

### Secondary (MEDIUM confidence)

- nvTorchCam paper (arXiv 2410.12074) — confirmed no refractive camera model; five-module architecture pattern; `diff_newton_inverse` precedent
- Refractive calibration tool (arXiv 2405.18018, ICCV 2025W) — three-module calibration component structure
- Kornia geometry docs — module organization pattern for geometry libraries
- OpenPTV docs — closest ecosystem analogue (C + Python bindings, different conventions, last active 2022)
- PyTorch compile Python 3.13 support — forum post cross-checked with release notes

### Tertiary (LOW confidence)

- NTNU refractive camera model — distortion model + virtual pinhole separation (single source, low community visibility)
- Oregon State Parrish group — n=1.333 is fresh water at 20°C approximation; seawater and temperature effects not modeled in v1

---
*Research completed: 2026-02-18*
*Ready for roadmap: yes*
