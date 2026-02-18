# Pitfalls Research

**Domain:** Refractive multi-camera geometry foundation library (PyTorch)
**Researched:** 2026-02-18
**Confidence:** HIGH — based on direct inspection of AquaCal and AquaMVS source codebases, confirmed bugs in test history, and verified physics/PyTorch behavior

---

## Critical Pitfalls

### Pitfall 1: Normal Vector Sign Convention Inversion in Snell's Law

**What goes wrong:**
The interface normal `[0, 0, -1]` points from water toward air (upward). When applying Snell's law using the vectorized formula `t = n_ratio * d + (cos_t - n_ratio * cos_i) * n`, `n` must point into the *destination* medium. For an air-to-water ray (going +Z downward), the oriented normal must be `[0, 0, +1]` (into water), not the stored `[0, 0, -1]`. Getting this wrong produces refracted rays that bend in the wrong direction — the angle increases instead of decreasing relative to the normal for air-to-water rays.

**Why it happens:**
The stored normal is a physical convention (from water toward air) used for Snell's law orientation checks, but the refraction formula requires the normal oriented *into* the destination medium. These are opposite signs. Code ported from a scalar formulation often handles this manually in conditionals that get dropped when vectorizing.

**How to avoid:**
Define and document the sign convention in a single place. In `snells_law_3d`, internally re-orient the normal based on `dot(incident, stored_normal)` sign before applying the formula. The AquaCal implementation does this correctly via `cos_i = -(rays_world * self.normal).sum(dim=-1)` with explicit negation; preserve this pattern exactly. Add a unit test for the canonical air-to-water case: a ray at 30° from vertical entering water must emerge at ~22° (Snell's: `sin(22°) = sin(30°) / 1.333`).

**Warning signs:**
- Triangulated points appear on the wrong side of the interface or at implausible depths.
- Round-trip test (project → back-project) shows > 1 mm error for points near the optical axis.
- Refracted ray Z-component is *smaller* than the incident ray Z-component after entering water.

**Phase to address:** Geometry foundation phase (Snell's law + `cast_ray`). Verify with a numeric known-value test: incident angle → exact expected refracted angle.

---

### Pitfall 2: Camera Center Derivation: C = -R^T @ t, Not -R @ t

**What goes wrong:**
The extrinsic convention is `p_cam = R @ p_world + t`, so the camera center in world coordinates is `C = -R^T @ t`. Using `-R @ t` instead produces a completely wrong camera center, causing projection, ray casting, and triangulation to fail silently — all functions accept the tensor without error, but output values are geometrically nonsense.

**Why it happens:**
`R` is the world-to-camera rotation. Inverting it for the world-space camera position requires the transpose `R^T`, not `R`. The two are identical only when `R = I` (no rotation), so the bug is invisible in unit tests that use cameras looking straight down with identity rotation.

**How to avoid:**
Always compute `C = -R.T @ t` (or `-(R.T @ t)`). Never use `-R @ t`. Document this formula in the `CameraExtrinsics` or equivalent dataclass. Add tests with tilted cameras (non-identity R) that compare `C` against a known world position.

**Warning signs:**
- Projection tests pass for cameras with `R = I` but fail when rotation is non-trivial.
- Camera ring visualization shows cameras at wrong world positions.
- Ray-plane intersection yields `t_param` outside the expected range.

**Phase to address:** Types and transforms phase. The `CameraExtrinsics.C` property must be defined correctly before any geometry can be tested.

---

### Pitfall 3: NumPy-to-PyTorch Migration Loses float64 Precision Where It Matters

**What goes wrong:**
The AquaCal implementation uses `float64` throughout. AquaMVS converts to `float32` for GPU efficiency. AquaCore must decide per-operation which precision is correct. Newton-Raphson for refractive projection converges to `1e-9` meters using `float64`; in `float32` the tolerance floor is around `1e-6` due to 7 decimal digits of precision. Triangulation with the linear least-squares system is numerically sensitive and can lose ~3 orders of magnitude of accuracy in `float32`. Silently downgrading to `float32` throughout produces wrong results that pass coarse tests.

**Why it happens:**
PyTorch defaults to `float32`. When porting NumPy (float64) code, `torch.tensor(numpy_array)` does NOT preserve float64 — it produces float32 by default. Developers assume parity with NumPy because the code looks identical.

**How to avoid:**
Use `torch.from_numpy(arr)` (which preserves dtype) rather than `torch.tensor(arr)` at all NumPy boundaries. For internal geometry math, explicitly annotate expected dtypes in function signatures and test that outputs match known float64 results at `atol=1e-6`. For the Newton-Raphson solver specifically, test convergence: the `r_p` solution should reproduce the original point to within `1e-4` m in float32 and `1e-9` m in float64. The AquaMVS implementation uses float32 with epsilon guards (`+ 1e-12`) to prevent div-by-zero — preserve these guards.

**Warning signs:**
- `torch.tensor(np.array(...))` produces `float32` — check with `assert tensor.dtype == torch.float64`.
- Round-trip accuracy regresses from `< 1e-6` (NumPy baseline) to `> 1e-3` (float32 precision loss).
- Newton-Raphson fails to converge in 10 iterations for points near the optical axis.

**Phase to address:** Geometry foundation phase. Establish dtype policy (float32 for GPU, float64 for precision-critical math) before implementing any math operations.

---

### Pitfall 4: In-Place Tensor Mutation Breaks Autograd Silently

**What goes wrong:**
The Newton-Raphson loop contains `r_p = torch.clamp(r_p, ...)` followed by `r_p = torch.minimum(r_p, r_q)`. If implemented as in-place operations (`r_p.clamp_()` or indexing `r_p[mask] = value`), PyTorch's autograd graph is corrupted. The backward pass either errors or silently produces zero gradients. The forward pass still produces correct numerical output, so the bug only surfaces when differentiating through the solver.

**Why it happens:**
In-place operations in a loop reuse the same tensor storage. PyTorch tracks in-place modifications via a version counter, and if a tensor is saved for backward before being modified in-place, the backward pass raises `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. When the error isn't raised (e.g., if `requires_grad=False`), gradients are wrong without any signal.

**How to avoid:**
Use out-of-place operations throughout the Newton-Raphson loop: `r_p = torch.clamp(r_p, min=0.0)` and `r_p = torch.minimum(r_p, r_q)`. The AquaMVS implementation already does this correctly. When porting, audit every `_=` in the loop. Add a test with `requires_grad=True` input: compute the Jacobian via `torch.autograd.functional.jacobian` and verify it matches a finite-difference estimate.

**Warning signs:**
- `RuntimeError: one of the variables needed for gradient computation` in backward pass.
- Gradients are all zero when `requires_grad=True` inputs flow through the solver.
- Test passes without `requires_grad` but fails when gradient tape is active.

**Phase to address:** Geometry foundation phase. If AquaCore will ever be used in a gradient-based optimizer (calibration refinement), this must be correct from the start. Even if not, it is a correctness invariant worth maintaining.

---

### Pitfall 5: Device Mismatch — Scalar Parameters Stranded on CPU When Tensors Move to CUDA

**What goes wrong:**
`RefractiveProjectionModel` stores `water_z` as a Python `float` and `n_air`, `n_water` as Python floats, while `K`, `R`, `t`, `C`, and `normal` are tensors. When `.to(device)` is called, the Python floats are not moved. In the Newton-Raphson loop, `h_c = self.water_z - C[2]` subtracts a CPU Python float from a CUDA tensor — this works in PyTorch (broadcasts), but `torch.full_like(px, self.water_z)` also works. The real failure mode is creating new tensors inside the loop via `torch.full((...), self.water_z)` without specifying `device=points.device`, which silently creates a CPU tensor that then participates in a CUDA computation and triggers a device error deep in the graph.

**Why it happens:**
The `to()` method only moves attributes that are explicitly listed. Python scalars are not tensors so they are not moved. Code that creates tensors from scalars mid-computation (e.g., `torch.tensor([self.water_z])`) will default to CPU regardless of model device.

**How to avoid:**
Audit every tensor creation inside `cast_ray` and `project`: all `torch.zeros`, `torch.ones`, `torch.full`, `torch.tensor` calls must include `device=points.device` and `dtype=points.dtype`. Use `torch.full_like` and `torch.zeros_like` where possible. In device-parametrized tests, run all geometry functions on both `cpu` and `cuda` (if available) and assert that output tensor devices match input tensor devices.

**Warning signs:**
- Tests pass on CPU but raise `RuntimeError: Expected all tensors to be on the same device` on CUDA.
- `torch.tensor(self.water_z, device=points.device)` appears inconsistently across methods.
- CI passes (CPU-only runner) but CUDA regression happens in production.

**Phase to address:** Geometry foundation phase, and must be caught by device-parametrized tests in the test suite before release.

---

### Pitfall 6: Offset Camera Regression — TIR Boundary Mistaken for Snell's Law Solution

**What goes wrong:**
The 1D Newton-Raphson search for the interface point clamps `r_p` to `[0, r_q]`. For cameras at nonzero XY offset from the target point, the initial pinhole guess overshoots: `r_p_init = r_q * h_c / (h_c + h_q)` can start outside `[0, r_q]` for large lateral offsets. After clamping, the iteration stalls at the boundary (`r_p = r_q`) which corresponds to a grazing-angle ray, not the Snell's law solution. The projection appears to succeed (no exception, no `NaN`) but the pixel is wrong. This exact bug existed in AquaCal's Brent-search path before the 2026-02-04 fix (see `TestOffsetCameraRoundTrip`).

**Why it happens:**
The initial guess formula assumes the interface point lies between the camera XY and the target XY, which is true for cameras centered above the target. For lateral-offset cameras, the geometry is different and the guess can exceed `r_q`. The clamp is a safeguard but also creates a false "converged" state.

**How to avoid:**
Test round-trip consistency (`project` → `back_project`) for cameras at several XY offsets (e.g., 0, 0.1 m, 0.3 m, 0.5 m). The tolerance must be `< 1e-4` m (the existing AquaCal standard). The Newton-Raphson implementation must verify convergence by checking `f(r_p) < tolerance` at the final value, not just `|delta| < tolerance`. If `f(r_p_final)` is large, the iteration stalled at a boundary — return `NaN`/invalid rather than the wrong pixel.

**Warning signs:**
- Round-trip error suddenly jumps from `~1e-9` to `~1e-1` m when camera XY offset exceeds ~0.2 m.
- `refractive_project` returns a pixel for a point that `refractive_back_project` cannot trace back to.
- Output pixel is near the image edge (large r_p corresponds to grazing-angle rays).

**Phase to address:** Geometry foundation phase. Include offset-camera round-trip tests explicitly.

---

### Pitfall 7: Depth Convention Confusion — Ray Depth vs. World Z

**What goes wrong:**
AquaCore defines depth as *ray depth* — distance along the refracted ray from the interface origin, not world Z-coordinate. The relationship is `point = origin + d * direction`. Code (or consumers) that conflates ray depth with world `Z - water_z` produces wrong 3D reconstructions: for oblique rays, `d * cos(theta_refracted)` is the world depth, not `d`. Mixing the two in the plane-sweep stereo or depth range computation produces silently incorrect results.

**Why it happens:**
For cameras looking straight down (no tilt, normal-incidence rays), `ray_depth ≈ world_depth` because `cos(theta) ≈ 1`. Tests with overhead cameras never expose the discrepancy. The distinction only surfaces for off-axis pixels or tilted cameras.

**How to avoid:**
Name every quantity precisely: `ray_depth` (along refracted ray), `water_depth` (`Z - water_z`). In docstrings, use: *"d is ray depth in meters along the refracted direction from the interface point; not the world-frame Z displacement."* Add a test for an oblique ray (30° from vertical) that verifies `origin + d * direction == point_3d` with the ray depth from `compute_depth_ranges`.

**Warning signs:**
- Depth range estimates pass with cameras looking straight down but fail with tilted cameras.
- `d_min` / `d_max` values are systematically off by a factor of `1/cos(theta)`.
- Consumer (AquaMVS plane sweep) produces blurred results only for peripheral pixels.

**Phase to address:** Geometry foundation phase. Document the convention in the module docstring and types.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy NumPy implementation as-is, wrap in `torch.from_numpy` at boundaries | Fast port, avoids rewrite | Mixed NumPy/PyTorch code path, CUDA path untested, dtype inconsistencies | Never — defeats purpose of AquaCore |
| Hardcode `n_water = 1.333` everywhere | Removes a parameter | Cannot model seawater (n≈1.341) or temperature variation; silently wrong at scale | Only in tests with explicit comment |
| Hardcode `device = "cpu"` in helper functions | Tests pass everywhere | Fails on CUDA with no warning; conceals device propagation bugs | Never |
| Use `numpy.ndarray` in intermediate geometry ops for "speed" | Avoids PyTorch overhead for small inputs | Breaks device-agnostic convention, forces CPU execution, breaks autograd | Never in public API |
| Skip back-projection test (only test projection) | Simpler test suite | Round-trip consistency (the only verifiable invariant for correct geometry) is not validated | Never — both directions must be tested |
| Reuse AquaCal's `Interface` dataclass via dependency | Avoids redefining types | Creates runtime dependency on AquaCal; consumers need both packages | Never — AquaCore must be standalone |

---

## Integration Gotchas

Common mistakes when connecting to external services (OpenCV, PyTorch linalg, calibration JSON).

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `torch.linalg.solve` (batched) | Assumes all systems are non-degenerate; error is raised for the entire batch if any row is singular | Pre-filter with `det.abs() > threshold` as done in `_triangulate_two_rays_batch`; fallback to per-element solve |
| `cv2.undistortPoints` | Returns normalized coordinates `(x/z, y/z)`, not pixel coordinates | Re-apply K to convert: `pixel = K[:2,:2] @ norm + K[:2,2]`; the AquaCal `undistort_points` function does this correctly |
| `cv2.fisheye.projectPoints` | Requires `D` shaped `(4, 1)`, not `(4,)` | Always `dist_coeffs.reshape(4, 1)` for fisheye path |
| OpenCV image size convention | `image_size = (width, height)` but NumPy arrays are `(height, width)` | Store as `(width, height)` in all dataclasses; swap to `(height, width)` only when creating NumPy arrays |
| `torch.from_numpy` | Shares memory with the numpy array; mutation of the numpy array corrupts the tensor | Call `.clone()` after `torch.from_numpy` when the numpy array may be modified externally |
| Calibration JSON loading | `water_z` may be stored per-camera in the JSON but should be the same for all cameras after optimization | Assert all per-camera `water_z` values are within 1 mm of each other at load time |

---

## Performance Traps

Patterns that work at small scale but degrade or fail at batch scale.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-point Python loop in Newton-Raphson (non-vectorized) | Projection of 10k points takes 30s instead of 0.1s | Use vectorized iteration as in `_refractive_project_newton_batch`; all N points iterate simultaneously | Any batch > ~1000 points on CPU |
| Calling `torch.linalg.solve` inside a Python for loop over ray pairs | Triangulating 100k pairs takes minutes | Use `_triangulate_two_rays_batch` with batched `torch.linalg.solve` | Any batch > ~100 pairs |
| Recomputing `K_inv = torch.linalg.inv(K)` on every `cast_ray` call | 3x slowdown in hot path | Precompute in `__init__` and store as `self.K_inv` | At dense stereo scale (millions of pixels) |
| Creating new `torch.eye(3)` inside batch triangulation loop | Memory allocation per call, no caching | Create once as `identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)` | Any non-trivial batch size |
| OpenCV `cv2.remap` called per-frame without precomputed maps | Remap table recomputed on each call | Precompute maps once with `compute_undistortion_maps`, cache the `UndistortionData` | From first call — always precompute |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Snell's law implementation:** Test with known angle pair (30° air → ~22° water) using exact trigonometry, not just "direction bends."
- [ ] **Newton-Raphson convergence:** Verify with `f(r_p_final) < tolerance`, not just `|delta| < tolerance`. A clamped-at-boundary result has wrong `f` but correct `|delta|`.
- [ ] **CUDA path:** All geometry tests pass on CPU, but device-parametrized tests (CUDA) have never run. Confirm CI has a CUDA runner or at minimum that all tensor creations include `device=` arguments.
- [ ] **Offset cameras (non-origin XY):** Round-trip `project` → `back_project` with cameras at 0.3 m XY offset — not just cameras at world origin.
- [ ] **Fisheye undistort path:** `cv2.fisheye.undistortPoints` is separate from `cv2.undistortPoints`; ensure `FisheyeCamera` calls the correct one.
- [ ] **`t` vector shape:** AquaCal serializes `t` as either `(3,)` or `(3, 1)` depending on how it was stored. The loader must handle both shapes (`squeeze()` before use).
- [ ] **Calibration loader standalone:** `load_calibration_data` in AquaCore must not import from `aquacal` at runtime. The JSON format must be parsed directly.
- [ ] **`water_z` scalar vs. tensor:** In `RefractiveProjectionModel.cast_ray`, `(self.water_z - self.C[2])` is float - tensor, which PyTorch allows, but `torch.full_like(px, self.water_z)` must verify `self.water_z` is a Python float, not a 0-dim tensor on the wrong device.
- [ ] **Round-trip at non-axis points:** Test points with both X and Y offsets from the camera's XY position, not just X-only or Y-only.
- [ ] **Batch batch-size=1:** Batch functions with N=1 input can silently broadcast incorrectly — test N=1 explicitly alongside N=100.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Sign error in normal convention | LOW | Flip sign in `snells_law_3d` normal orientation; re-run tests; no API change required |
| Camera center formula wrong (-R@t not -R^T@t) | LOW | Fix formula in `CameraExtrinsics.C` property; all downstream code auto-corrects |
| float64→float32 precision loss throughout | MEDIUM | Add dtype parameter to internal math functions; update tests to assert dtype; no API break if `torch.from_numpy` is used at boundaries |
| In-place mutation breaking autograd | LOW | Replace `tensor[mask] = value` and `tensor.clamp_()` with out-of-place equivalents; no logic change |
| Device mismatch discovered in CUDA testing | LOW-MEDIUM | Audit all tensor creation calls in affected functions; add `device=` args; covered by device-parametrized tests |
| Offset camera bug (stalled at TIR boundary) | MEDIUM | Add convergence check on `f(r_p_final)` and return `NaN` on non-convergence; add offset-camera round-trip test as regression |
| AquaCore accidentally imports AquaCal at runtime | HIGH | Refactor loader to parse JSON directly; update all tests to avoid AquaCal; potentially requires new JSON schema constants |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Normal sign convention | Geometry foundation (Snell's law) | Known-value Snell's law test: sin(θ₁)/sin(θ₂) = n₂/n₁ |
| Camera center formula | Types and transforms phase | Non-identity R test: camera at known world position |
| float64 precision loss | Geometry foundation phase | Round-trip atol=1e-6 m with known points |
| In-place autograd corruption | Geometry foundation phase | `requires_grad=True` Jacobian test |
| Device mismatch | Geometry foundation phase | Device-parametrized pytest fixture (cpu + cuda) |
| Offset camera TIR stall | Geometry foundation phase | Round-trip test grid over camera XY offsets |
| Depth convention confusion | Geometry foundation phase | Oblique ray: verify `origin + d * dir == point_3d` |
| Calibration loader coupling | I/O and calibration loader phase | Import test: `import aquacore` with AquaCal uninstalled |
| Batch size=1 broadcasting | Geometry foundation phase | Explicit N=1 test case for all batch functions |
| OpenCV fisheye D shape | Camera models phase | Fisheye project/back-project round-trip |

---

## Sources

- AquaCal source: `../AquaCal/src/aquacal/core/refractive_geometry.py` — contains Newton-Raphson and Brent-search implementations, the n_ratio sign convention, and offset-camera regression tests
- AquaCal source: `../AquaCal/src/aquacal/core/interface_model.py` — Interface normal convention documentation
- AquaCal tests: `../AquaCal/tests/unit/test_refractive_geometry.py` — `TestOffsetCameraRoundTrip` class documents the 2026-02-04 TIR boundary bug fix; `test_round_trip_multiple_offset_cameras` is the regression guard
- AquaMVS source: `../AquaMVS/src/aquamvs/projection/refractive.py` — PyTorch Newton-Raphson with out-of-place clamp; `cos_i` negation pattern for air-to-water rays
- AquaMVS source: `../AquaMVS/src/aquamvs/triangulation.py` — batched triangulation with determinant filter; `_triangulate_two_rays_batch`
- AquaMVS source: `../AquaMVS/src/aquamvs/calibration.py` — `t_numpy.ndim == 2` shape guard; `torch.from_numpy` dtype preservation pattern
- PyTorch docs: [Autograd mechanics — in-place operations](https://docs.pytorch.org/docs/stable/notes/autograd.html) — official statement discouraging in-place ops
- PyTorch docs: [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — float32 precision limits
- Research: ["Analysis of the Influence of Refraction-Parameter Deviation on Underwater Stereo-Vision"](https://www.mdpi.com/2072-4292/16/17/3286) — error analysis for calibration deviations
- Research: Oregon State Parrish group — refractive index of water as function of temperature, salinity, wavelength (n=1.333 is approximation for fresh water at 20°C, visible light)
- OpenCV: [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — `p_cam = R @ p_world + t` convention; C = -R^T @ t derivation
- Computer vision conventions blog: ["Camera Conventions, Transforms, and Conversions"](https://blog.mkari.de/posts/cam-transform/) — 2,556 ways to get transforms wrong; duality of pose vs. change-of-coordinates

---
*Pitfalls research for: Refractive multi-camera geometry foundation library (AquaCore)*
*Researched: 2026-02-18*
