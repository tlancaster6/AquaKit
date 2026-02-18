# Stack Research

**Domain:** Refractive multi-camera geometry foundation library (Python/PyTorch)
**Researched:** 2026-02-18
**Confidence:** HIGH (core stack verified via PyPI + official docs; domain-specific patterns MEDIUM)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | >=3.11 | Runtime | `X \| Y` union syntax, `match` statements, `tomllib` stdlib. 3.11 is the oldest supported release in the CI matrix; 3.12 and 3.13 are tested too. PyTorch 2.10 supports all three. |
| PyTorch (`torch`) | >=2.6, pin to 2.10 at dev time | All tensor math: Snell's law, projection, triangulation, rotation ops | Only framework with batched GPU-accelerated linalg, autograd, and `torch.compile` in a single package. `torch.linalg.svd` supports batched (B, M, N) inputs natively — critical for batched triangulation. 2.10 is current stable (released 2026-01-21). |
| OpenCV (`opencv-python-headless`) | >=4.11, pin to 4.13.0.92 at dev time | Lens undistortion map computation (`cv2.initUndistortRectifyMap`), image I/O, fisheye undistortion | Only library that implements the full OpenCV distortion model (radial k1-k6, tangential p1-p2, thin prism, tilt) with C-speed. `headless` variant avoids Qt/X11 dependency chain — no GUI needed in a server library. |
| Hatch / Hatchling | hatch (any), hatchling >=1.28 | Build system, env management, dev workflow | Already configured in `pyproject.toml`. PEP 517/518 compliant, pure Python, zero Rust toolchain required. Hatchling 1.28.0 is current stable (2025-11-27). Project does NOT need uv_build — that is the better choice only for new projects starting from scratch with uv as the package manager, not for existing Hatch projects. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| kornia | >=0.8.2 | Differentiable image ops: color conversion, image filtering, warp affine, augmentation | Optional — use for image-level ops that would otherwise require writing custom CUDA kernels. Do NOT use kornia's camera geometry classes (`PinholeCamera`) as AquaCore has its own typed models. kornia 0.8.2 is current stable (2025-11-08). |
| NumPy | >=1.26 (pulled in by PyTorch transitively) | Serialization boundary only: JSON→ndarray→torch.tensor, OpenCV call inputs/outputs | Only at I/O and calibration load boundaries. Never use NumPy for geometry math — keep all intermediate computation in PyTorch. |
| jaxtyping | >=0.2.35 | Shape + dtype annotations on public API (`Float[Tensor, "B 3"]`) | Use in all public function signatures where tensor shape is semantically meaningful. Pairs with beartype for optional runtime checking. Recommended over torchtyping (abandoned) and Annotated-shape hacks. Latest stable: ~0.2.38 (verify on PyPI before pinning). |
| beartype | >=0.19 | Runtime type enforcement at development and test time | Optional, pairs with jaxtyping. Wrap test functions with `@beartype` to catch shape errors automatically. Do NOT add as a production dependency — keep optional or dev-only. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Ruff | Linter + formatter | Already configured. Current version 0.15.1 (2026-02-12). Rules E4/E7/E9, F, W, I, UP, B, SIM, RUF, PT already selected in `pyproject.toml` — no changes needed. |
| basedpyright | Static type checker | Already configured in basic mode. Current version 1.38.1 (2026-02-18). Stricter than upstream pyright; adds `reportAny` and `reportUnreachable`. Basic mode is the right call for a library with PyTorch (avoid false positives from Any-typed torch internals). |
| pytest | >=8.3, pin to 9.0.2 at dev time | Test runner | Current stable 9.0.2 (2025-12-06). Use `pytest.mark.parametrize` with `["cpu", "cuda"]` device strings rather than pytest-pytorch plugin (fewer dependencies, simpler). Skip CUDA tests with `pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`. |
| pytest-cov | >=6.0 | Coverage measurement | Already in dev deps. Integrate with Codecov via existing `.github/workflows/test.yml`. |
| pre-commit | >=4.0 | Git hook runner | Already configured. `.pre-commit-config.yaml` exists. |
| detect-secrets | managed via pre-commit | Secret scanning | Already in `.secrets.baseline`. Regenerate after source file changes. |

---

## Installation

```bash
# Core runtime (pyproject.toml [project] dependencies)
# Add these to pyproject.toml:
torch>=2.6
opencv-python-headless>=4.11

# Optional (consumers opt in):
kornia>=0.8

# Dev dependencies (already in [tool.hatch.envs.default])
pytest>=8.3
pytest-cov>=6.0
ruff>=0.15
pre-commit>=4.0
basedpyright>=1.38

# Optional dev extras for tensor shape checking:
jaxtyping>=0.2.35
beartype>=0.19
```

Full pyproject.toml `dependencies` block (for AquaCore as a distributable library):
```toml
[project]
dependencies = [
    "torch>=2.6",
    "opencv-python-headless>=4.11",
]

[project.optional-dependencies]
kornia = ["kornia>=0.8"]
typing = ["jaxtyping>=0.2.35", "beartype>=0.19"]
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `opencv-python-headless` | `opencv-python` | Only if consumers need OpenCV GUI (`cv2.imshow`). Library itself never needs GUI. |
| `torch.linalg.svd` (batched) | scipy.linalg, numpy.linalg | Never — NumPy/SciPy do not run on GPU and break the PyTorch-first contract. |
| `jaxtyping` for shape annotations | `torchtyping` | Never use torchtyping — abandoned since 2023, does monkey-patching, incompatible with modern Python. |
| `jaxtyping` for shape annotations | Plain `torch.Tensor` with docstring shapes | Acceptable if jaxtyping is too heavy a dependency signal. jaxtyping is preferred for public APIs. |
| `kornia` (optional) | Custom CUDA kernels | If kornia op is unavailable; but exhaust kornia first — it is already a declared optional dep. |
| `kornia` (optional) | PyTorch3D | Never for this domain — PyTorch3D is a rendering-focused library, lacks undistortion, adds >500 MB of compiled CUDA extensions. Violates the "no heavy ML deps" constraint. |
| `beartype` (optional dev) | `typeguard` | Either works; beartype is faster (O(1) checks), preferred in PyTorch ecosystem with jaxtyping. |
| Hatch | uv | uv_build is better for new projects using uv as package manager. AquaCore is already Hatch-configured — switching would break existing CI/CD and provide no meaningful benefit. |
| basedpyright (basic mode) | mypy | basedpyright is faster, already configured, has better PyTorch stub support via `pytorch-stubs`. mypy requires separate plugin for torch and has slower incremental checking. |
| pytest parametrize devices | pytest-pytorch plugin | pytest-pytorch adds a dependency for functionality achievable with 3 lines of pytest marks. Avoid unnecessary deps in a library. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `Open3D` | Pulls in heavy compiled extensions (>500 MB), has its own tensor system that conflicts with PyTorch's device model. Explicitly called out in project constraints as "stays in consumers". | Implement point-cloud geometry directly with `torch.linalg` |
| `LightGlue` / `RoMa` | Feature matching ML models — wrong abstraction level for a geometry foundation library. | None — not needed |
| `PyTorch3D` | Rendering-focused, massive compile footprint, unnecessary for camera math. Does not model refractive interfaces. | `torch.linalg` + kornia for image ops |
| `torchtyping` | Abandoned project, monkey-patches `torch.Tensor`, incompatible with Python 3.11+ strict typing. | `jaxtyping` |
| `opencv-python` (full) | Pulls in Qt/X11 libraries — conflict risk in headless CI and server environments, larger install. | `opencv-python-headless` |
| NumPy math for geometry | Breaks GPU execution, forces CPU round-trips, creates device/dtype mismatch bugs. | `torch` everywhere; NumPy only at JSON boundary |
| `scipy` for linalg | CPU-only, requires array conversion. Already available via PyTorch. | `torch.linalg.solve`, `torch.linalg.svd`, etc. |
| `nvTorchCam` | Does not model refractive interfaces. Useful for pinhole/fisheye camera abstraction in rendering contexts but does not address Snell's law or air-water refraction. | Implement `RefractiveProjectionModel` directly |

---

## Stack Patterns by Variant

**If geometry op requires iterative solving (e.g., Newton-Raphson back-projection):**
- Implement as `torch.autograd.Function` with custom `backward()` using the implicit function theorem
- The forward pass runs the iterator; backward computes `dL/dx = dL/dy * (dy/dx)^{-1}` analytically
- Do NOT use `torch.func.jacrev` inside the iteration loop — avoid materializing the Jacobian per-step

**If a function must be both CPU and CUDA correct:**
- Write once in pure PyTorch ops (no device-specific branches)
- Parametrize tests: `@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])`
- Never hardcode `.cuda()` or `.cpu()` in library code — follow tensor device

**If kornia is available (optional dep):**
- Use `kornia.geometry.transform.warp_perspective` for image warping
- Use `kornia.enhance.*` for color-space conversion at I/O boundary
- Do NOT use `kornia.geometry.camera.PinholeCamera` — AquaCore defines its own typed `Camera` model

**If tensor shape annotation is desired (optional typing extras):**
- Use `jaxtyping.Float[torch.Tensor, "B 3"]` in function signatures
- Wrap with `@beartype` in test files to get runtime checking during `pytest`
- Do NOT make `jaxtyping`/`beartype` a hard runtime dep — keep in `[project.optional-dependencies]`

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `torch==2.10.0` | Python 3.10–3.13 | `torch.compile` fully supports Python 3.11, 3.12, 3.13 as of 2.6/2.7/2.9 respectively |
| `torch>=2.6` | `kornia>=0.8.0` | kornia 0.8.0 (2025-01-11) dropped support for torch < 2.0; 0.8.2 tested against torch 2.x |
| `opencv-python-headless>=4.11` | All Python 3.7–3.14 | No known conflicts with PyTorch. Do NOT install `opencv-python` alongside `opencv-python-headless` in the same env — they share `cv2` namespace and conflict. |
| `basedpyright>=1.38` | `torch` (via `pytorch-stubs`) | basedpyright resolves torch stubs from `torch`'s bundled `py.typed` marker. No separate stub package needed for torch 2.x. |
| `jaxtyping>=0.2.35` | Python >=3.10, `torch` any | jaxtyping is framework-agnostic — works with torch, numpy, jax. No JAX runtime required. |

---

## Sources

- **PyPI: torch** — https://pypi.org/project/torch/ — Version 2.10.0 confirmed current stable (2026-01-21). HIGH confidence.
- **PyPI: opencv-python-headless** — https://pypi.org/project/opencv-python-headless/ — Version 4.13.0.92 confirmed (2026-02-05). HIGH confidence.
- **PyPI: kornia** — https://pypi.org/project/kornia/ — Version 0.8.2 confirmed (2025-11-08). HIGH confidence.
- **PyPI: ruff** — https://pypi.org/project/ruff/ — Version 0.15.1 confirmed (2026-02-12). HIGH confidence.
- **PyPI: basedpyright** — https://pypi.org/project/basedpyright/ — Version 1.38.1 confirmed (2026-02-18). HIGH confidence.
- **PyPI: pytest** — https://pypi.org/project/pytest/ — Version 9.0.2 confirmed (2025-12-06). HIGH confidence.
- **PyPI: hatchling** — https://pypi.org/project/hatchling/ — Version 1.28.0 confirmed (2025-11-27). HIGH confidence.
- **kornia geometry.camera docs** — https://kornia.readthedocs.io/en/latest/geometry.camera.html — Confirmed Kannala-Brandt distortion support, no refractive interface model. MEDIUM confidence.
- **nvTorchCam paper** — https://arxiv.org/html/2410.12074v1 — Confirmed no refractive camera model support. MEDIUM confidence.
- **PyTorch torch.linalg docs** — https://docs.pytorch.org/docs/stable/linalg.html — Confirmed batched SVD, solve; stable API since torch 1.9. HIGH confidence.
- **PyTorch compile Python 3.13 support** — https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-13-completed/2738 — MEDIUM confidence (forum post, cross-checks with torch 2.9 release notes).
- **jaxtyping recommendation over torchtyping** — https://kidger.site/thoughts/jaxtyping/ — Author of both libraries recommends jaxtyping. HIGH confidence for deprecation of torchtyping.
- **opencv-python-headless vs opencv-python** — https://github.com/opencv/opencv-python — Official project explains headless variant. HIGH confidence.
- **PyPI: jaxtyping** — Latest ~0.2.38; verify at https://pypi.org/project/jaxtyping/ before pinning. MEDIUM confidence on exact version (PyPI page returned error, version from WebSearch cross-check).

---

*Stack research for: AquaCore — refractive multi-camera geometry foundation library*
*Researched: 2026-02-18*
