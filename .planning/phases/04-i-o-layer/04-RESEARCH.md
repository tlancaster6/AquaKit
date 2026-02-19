# Phase 4: I/O Layer - Research

**Researched:** 2026-02-18
**Domain:** Synchronized multi-camera frame I/O — OpenCV video/image reading, Python Protocol, PyTorch tensor conversion
**Confidence:** HIGH — research drawn from AquaMVS reference implementation (readable on this machine) and existing AquaCore patterns

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Frame tensor format:**
- Layout: **(C, H, W) float32** — PyTorch convention, works natively with torchvision and conv layers
- Color order: **RGB** — BGR-to-RGB conversion happens inside the I/O layer; consumers get standard RGB
- Value range: Claude's discretion (see below)
- Output type: **PyTorch tensors only** — no NumPy return option; consumers call `.numpy()` if needed
- Independent copies: tensors must be `.clone()`d from OpenCV buffers to prevent silent overwrite on next read

**Access & iteration API:**
- FrameSet protocol defines **`__getitem__`**, **`__len__`**, and **`__iter__`**
- `__getitem__(idx)` returns `dict[str, Tensor]` — camera name keys, (C, H, W) float32 tensor values
- `__len__` returns total frame count
- `__iter__` yields `(frame_idx, dict[str, Tensor])` tuples — sequential, frame-exact
- **VideoSet caveat:** `__getitem__` uses cv2 seek (approximate for compressed video — fine for locating window starts); `__iter__` reads sequentially and is frame-exact
- ImageSet: all access is exact (file-based)
- FrameSet protocol **requires context manager** (`__enter__`/`__exit__`); ImageSet is a no-op, VideoSet releases cv2.VideoCapture handles

**Camera-to-path mapping:**
- Constructor takes **`dict[str, str | Path]`** — explicit camera name to file/directory path
- Internally converts all paths to `Path` objects
- One video file per camera (no multi-camera-in-one-file support)
- **Factory function** `create_frameset(camera_map)` auto-detects images vs video from paths and returns the appropriate concrete class
- No CalibrationData coupling — camera-to-path mapping is established by consumer repos (AquaCal init, AquaMVS init), not AquaCore

**Frame mismatch & errors:**
- **Mismatched frame counts:** warn and use minimum count (not ValueError like AquaMVS)
- **Corrupt/unreadable frames:** warn + omit that camera from the returned dict (AquaMVS pattern); consumer checks if expected cameras are present
- **ImageSet filename matching:** require matching filenames across all camera dirs (sorted order), raise ValueError on mismatch — catches data alignment issues early
- **Missing directories/files:** raise ValueError at init (same as AquaMVS)

### Claude's Discretion

- Float32 value range ([0, 1] vs [0, 255]) — pick based on typical consumer patterns
- Exact image extensions supported (png, jpg, tiff, etc.)
- Logging verbosity and format
- Internal buffering strategy for VideoSet sequential reads

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 4 delivers three files that are currently empty stubs: `frameset.py` (the Protocol), `video.py` (VideoSet), and `images.py` (ImageSet), plus a factory function and updates to `io/__init__.py` and the top-level `__init__.py`. The work is primarily a port of AquaMVS's `ImageDirectorySet` extended with VideoSet and a common Protocol, with specific divergences: (1) the output format is `(C, H, W) float32` PyTorch tensors instead of `(H, W, 3) uint8` NumPy arrays; (2) the API is `__getitem__`/`__len__`/`__iter__` instead of `read_frame`/`iterate_frames`; (3) the Protocol requires context manager support (`__enter__`/`__exit__`).

The reference implementation (`AquaMVS/src/aquamvs/io.py`) is fully readable on this machine and provides verified patterns for directory globbing, sorted filename matching, cv2.imread error handling, cv2.VideoCapture seek/sequential reading, and `detect_input_type()` factory logic. No new library dependencies are needed — `opencv-python` and `torch` are already in `pyproject.toml`.

The only Claude's discretion area that significantly affects architecture is **float32 value range**. The recommendation is **[0, 1]** (divide by 255.0), matching torchvision convention and consumer expectations for neural-network inputs. The conversion path is `torch.from_numpy(bgr_uint8[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0`, which handles BGR→RGB flip, (H,W,C)→(C,H,W) permute, dtype conversion, and normalization in a single chain.

**Primary recommendation:** Port AquaMVS's `ImageDirectorySet` into `ImageSet` with Protocol compliance, implement `VideoSet` using `cv2.VideoCapture` with sequential-read optimization for `__iter__`, define `FrameSet` as a `runtime_checkable` Protocol mirroring the `ProjectionModel` pattern already established in Phase 2, and expose `create_frameset` as the public factory. No new dependencies. No custom file-reading logic.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| opencv-python | >=4.8 | cv2.imread, cv2.VideoCapture — only library providing both video seek and image directory reading | Already in pyproject.toml; the AquaMVS reference uses it directly |
| torch | >=2.0 | Tensor output format, `.permute()`, `.float()`, `.clone()` | Project-wide requirement; the output contract is PyTorch tensors |
| pathlib.Path | stdlib | Path normalization in constructors | Used uniformly across all AquaCore modules |
| warnings | stdlib | Warn on frame count mismatch, corrupt frames | Already chosen in Phase 3 for library warnings; consistent |
| logging | stdlib | Operational info messages (frame counts, camera counts) | Already used in AquaMVS io.py; appropriate for debug/info messages |
| typing.Protocol, runtime_checkable | stdlib | FrameSet protocol definition | Same pattern as Phase 2's ProjectionModel; consistent |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.24 | Intermediate buffer for cv2 output before torch conversion | Only at the OpenCV boundary; no NumPy stored on the class |
| Iterator, Generator from typing | stdlib | Return type annotations for `__iter__` | Type-correct iteration signatures |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cv2.VideoCapture | PyAV, imageio, decord | cv2 is already a project dependency; other libraries add deps; cv2 is sufficient for sequential read + seek use cases |
| `[0, 1]` float32 range | `[0, 255]` float32 range | [0,1] matches torchvision, standard for neural network inputs. [0,255] saves one division but breaks consumer expectations. Recommendation: [0,1]. |
| `warnings.warn` for frame mismatch | `logging.warning` | `warnings.warn` is for data quality issues detected at construction time (one-time alerts). `logging.warning` is for runtime events. Use `warnings.warn` for init-time mismatch, `logging.warning` for per-frame read failures. |

**Installation:** No new dependencies. `torch`, `numpy`, `opencv-python` are already in `pyproject.toml`.

---

## Architecture Patterns

### Module-to-File Mapping

Phase 4 fills three stubs plus updates two `__init__.py` files:

```
src/aquacore/
└── io/
    ├── __init__.py     # Exports: FrameSet, VideoSet, ImageSet, create_frameset
    ├── frameset.py     # FrameSet Protocol (runtime_checkable)
    ├── video.py        # VideoSet implementation
    └── images.py       # ImageSet implementation + create_frameset factory
```

Top-level `src/aquacore/__init__.py` adds FrameSet, VideoSet, ImageSet, create_frameset to public API.

### Pattern 1: FrameSet as runtime_checkable Protocol

**What:** `FrameSet` is a `typing.Protocol` decorated with `@runtime_checkable` — exactly the same pattern as `ProjectionModel` from Phase 2 (`src/aquacore/projection/protocol.py`). Concrete classes implement the protocol structurally without inheriting from it.

**When to use:** Always — this is the locked design. The Protocol enables duck typing: code written against `FrameSet` type-checks with both `VideoSet` and `ImageSet` without modification.

**Key protocol methods:**

```python
# Source: Phase 2 ProjectionModel pattern (projection/protocol.py) + locked decisions
from typing import Iterator, Protocol, runtime_checkable
import torch

@runtime_checkable
class FrameSet(Protocol):
    """Protocol for synchronized multi-camera frame access.

    Any class implementing __getitem__, __len__, __iter__, __enter__, and
    __exit__ with the correct signatures satisfies this protocol structurally.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return frame by index.

        Args:
            idx: Zero-based frame index.

        Returns:
            Mapping from camera name to (C, H, W) float32 tensor in [0, 1].
            Missing cameras (unreadable frames) are omitted from the dict.
        """
        ...

    def __len__(self) -> int:
        """Return total frame count (minimum across all cameras for VideoSet)."""
        ...

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate frames sequentially, yielding (frame_idx, dict[str, Tensor])."""
        ...

    def __enter__(self) -> "FrameSet":
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...
```

### Pattern 2: ImageSet — Direct Port of AquaMVS ImageDirectorySet

**What:** ImageSet validates directories and builds a sorted filename index at construction. `__getitem__` and `__iter__` both use `cv2.imread` + BGR→RGB conversion + tensor conversion. Context manager is a no-op. Strict filename matching enforced at init.

**When to use:** When all camera paths are directories containing image files.

**Verified reference:** `AquaMVS/src/aquamvs/io.py` `ImageDirectorySet` class — readable on this machine.

**Key divergences from AquaMVS reference:**
1. Returns `(C, H, W) float32 [0,1]` tensors instead of `(H, W, 3) uint8` NumPy arrays
2. `__getitem__`/`__len__`/`__iter__` interface instead of `read_frame`/`iterate_frames`
3. `__iter__` yields `(frame_idx, dict)` tuples instead of `(frame_idx, images)` with different types
4. Frame count mismatch: warn + use minimum (not ValueError) — but filename mismatch still raises ValueError

```python
# Source: AquaMVS/src/aquamvs/io.py ImageDirectorySet (adapted)
import warnings
import logging
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp")

class ImageSet:
    """Synchronized frame access from per-camera image directories.

    Constructor validates directory existence, globs image files for each camera,
    enforces matching filenames across cameras, and builds the frame index.
    """

    def __init__(self, camera_map: dict[str, str | Path]) -> None:
        self._dirs = {name: Path(p) for name, p in camera_map.items()}
        self._frame_files: dict[str, list[Path]] = {}
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        for cam_name, cam_dir in self._dirs.items():
            if not cam_dir.exists():
                raise ValueError(f"Camera directory does not exist: {cam_dir}")
            if not cam_dir.is_dir():
                raise ValueError(f"Camera path is not a directory: {cam_dir}")

            files: list[Path] = []
            for ext in _IMAGE_EXTENSIONS:
                files.extend(cam_dir.glob(ext))
            if not files:
                raise ValueError(f"No images found in: {cam_dir}")

            self._frame_files[cam_name] = sorted(files, key=lambda p: p.name)

        # Check filename consistency (raise on mismatch — catches alignment issues)
        reference_cam = next(iter(self._frame_files))
        reference_names = [f.name for f in self._frame_files[reference_cam]]
        for cam_name, files in self._frame_files.items():
            names = [f.name for f in files]
            if names != reference_names:
                raise ValueError(
                    f"Filenames do not match between '{reference_cam}' and '{cam_name}'."
                )

        # Frame count mismatch: warn and use minimum (not ValueError)
        counts = {n: len(f) for n, f in self._frame_files.items()}
        if len(set(counts.values())) != 1:
            min_count = min(counts.values())
            warnings.warn(
                f"Frame counts differ across cameras: {counts}. "
                f"Using minimum: {min_count}.",
                stacklevel=2,
            )
            for cam_name in self._frame_files:
                self._frame_files[cam_name] = self._frame_files[cam_name][:min_count]

        self._frame_count = min(len(f) for f in self._frame_files.values())
        logger.info(
            "ImageSet: %d frames, %d cameras",
            self._frame_count,
            len(self._dirs),
        )

    def _read_frame_dict(self, idx: int) -> dict[str, torch.Tensor]:
        """Read one frame index from all cameras, returning (C, H, W) float32 tensors."""
        result: dict[str, torch.Tensor] = {}
        for cam_name, files in self._frame_files.items():
            bgr = cv2.imread(str(files[idx]))
            if bgr is None:
                warnings.warn(
                    f"Failed to read image: {files[idx]} (camera '{cam_name}', frame {idx})",
                    stacklevel=3,
                )
                continue
            # BGR (H,W,3) uint8 -> RGB (H,W,3) uint8 -> (C,H,W) float32 [0,1]
            rgb = bgr[..., ::-1].copy()          # BGR->RGB (copy avoids negative stride)
            t = torch.from_numpy(rgb)             # (H, W, 3) uint8
            t = t.permute(2, 0, 1).float() / 255.0  # (C, H, W) float32 [0,1]
            result[cam_name] = t
        return result

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        return self._read_frame_dict(idx)

    def __len__(self) -> int:
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        for idx in range(self._frame_count):
            yield idx, self._read_frame_dict(idx)

    def __enter__(self) -> "ImageSet":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # No resources to release
```

### Pattern 3: VideoSet — cv2.VideoCapture with Sequential Optimization

**What:** VideoSet opens one `cv2.VideoCapture` per camera at construction. `__getitem__` uses `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` for seek (approximate for compressed video). `__iter__` reads sequentially using `cap.read()` which is frame-exact for all codecs.

**When to use:** When all camera paths are video files.

**No AquaMVS VideoSet reference exists** — AquaMVS only has `ImageDirectorySet`. VideoSet must be implemented fresh using cv2.VideoCapture patterns.

**Frame count mismatch:** warn and use minimum count (same as ImageSet mismatch handling). Computed from `cap.get(cv2.CAP_PROP_FRAME_COUNT)` at init.

**Context manager:** `__exit__` must call `cap.release()` for each capture object. This is the primary reason the Protocol requires context manager — VideoCapture holds OS file handles.

```python
# Source: cv2.VideoCapture API patterns (standard OpenCV usage)
import warnings
import logging
from pathlib import Path
from typing import Iterator
import cv2
import torch

logger = logging.getLogger(__name__)


class VideoSet:
    """Synchronized frame access from per-camera video files.

    __iter__ is frame-exact (sequential read). __getitem__ uses cv2 seek
    which is approximate for compressed video (keyframe-based) — suitable
    for locating temporal windows, not for precise frame retrieval.

    Must be used as a context manager to release VideoCapture handles.
    """

    def __init__(self, camera_map: dict[str, str | Path]) -> None:
        self._paths = {name: Path(p) for name, p in camera_map.items()}
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._frame_count: int = 0
        self._open_captures()

    def _open_captures(self) -> None:
        for cam_name, path in self._paths.items():
            if not path.exists():
                raise ValueError(f"Video file does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Video path is not a file: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {path}")
            self._caps[cam_name] = cap

        # Frame counts per camera
        counts = {
            name: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for name, cap in self._caps.items()
        }

        if len(set(counts.values())) != 1:
            min_count = min(counts.values())
            warnings.warn(
                f"Video frame counts differ across cameras: {counts}. "
                f"Using minimum: {min_count}.",
                stacklevel=2,
            )
            self._frame_count = min_count
        else:
            self._frame_count = next(iter(counts.values()))

        logger.info(
            "VideoSet: %d frames, %d cameras",
            self._frame_count,
            len(self._caps),
        )

    def _bgr_to_tensor(self, bgr: "np.ndarray") -> torch.Tensor:
        """Convert (H, W, 3) uint8 BGR ndarray to (C, H, W) float32 [0,1] tensor."""
        rgb = bgr[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Read frame by index using cv2 seek (approximate for compressed video)."""
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        result: dict[str, torch.Tensor] = {}
        for cam_name, cap in self._caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                warnings.warn(
                    f"Failed to read frame {idx} from camera '{cam_name}'",
                    stacklevel=2,
                )
                continue
            result[cam_name] = self._bgr_to_tensor(bgr)
        return result

    def __len__(self) -> int:
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate frames sequentially — frame-exact for all codecs."""
        # Reset all captures to frame 0 for sequential read
        for cap in self._caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for idx in range(self._frame_count):
            result: dict[str, torch.Tensor] = {}
            for cam_name, cap in self._caps.items():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    warnings.warn(
                        f"Failed to read frame {idx} from camera '{cam_name}'",
                        stacklevel=2,
                    )
                    continue
                result[cam_name] = self._bgr_to_tensor(bgr)
            yield idx, result

    def __enter__(self) -> "VideoSet":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        logger.debug("VideoSet: released %d captures", len(self._paths))
```

### Pattern 4: create_frameset Factory — Port of detect_input_type

**What:** `create_frameset(camera_map)` detects whether paths are directories or video files and returns the appropriate concrete class. Logic is ported from AquaMVS's `detect_input_type()` (verified in `AquaMVS/src/aquamvs/io.py`).

**When to use:** When the caller does not know in advance whether they have images or video.

```python
# Source: AquaMVS/src/aquamvs/io.py detect_input_type() (adapted — returns instance, not string)
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}

def create_frameset(camera_map: dict[str, str | Path]) -> "ImageSet | VideoSet":
    """Auto-detect input type and return the appropriate FrameSet.

    Args:
        camera_map: Mapping from camera name to file or directory path.

    Returns:
        ImageSet if paths are directories; VideoSet if paths are video files.

    Raises:
        ValueError: If camera_map is empty or contains mixed types.
    """
    if not camera_map:
        raise ValueError("camera_map must not be empty")

    paths = [Path(p) for p in camera_map.values()]
    is_dir = [p.is_dir() for p in paths]
    is_file = [p.is_file() for p in paths]

    if all(is_dir):
        return ImageSet(camera_map)
    if all(is_file):
        return VideoSet(camera_map)
    # Paths don't exist yet — infer from extension (test/mock paths)
    if not any(is_dir) and not any(is_file):
        has_video_ext = [p.suffix.lower() in _VIDEO_EXTENSIONS for p in paths]
        if all(has_video_ext):
            return VideoSet(camera_map)
        if not any(has_video_ext):
            return ImageSet(camera_map)
    raise ValueError(
        "camera_map contains mixed types (directories and files). "
        "All paths must be either directories or files."
    )
```

### Pattern 5: BGR to (C, H, W) float32 [0, 1] Tensor Conversion

**What:** The critical conversion path from OpenCV output to AquaCore's tensor format. This is not in AquaMVS (which returns NumPy) so it must be implemented fresh.

**Steps:**
1. `bgr[..., ::-1]` — reverse channel axis: BGR→RGB. This creates a **view with a negative stride**.
2. `.copy()` — materialize the view to a contiguous array. Required because `torch.from_numpy` requires contiguous memory (negative stride fails).
3. `torch.from_numpy(rgb)` — zero-copy wrap as `(H, W, 3)` uint8 tensor (shares memory with the `.copy()` result).
4. `.permute(2, 0, 1)` — reorder to `(C, H, W)`. Returns a non-contiguous view.
5. `.float()` — cast to float32 (implicit copy, result is contiguous).
6. `/ 255.0` — normalize to `[0, 1]`.

**Why `.clone()` is NOT needed here:** The `/ 255.0` operation creates a new tensor (it is not in-place). The result does not share memory with the OpenCV buffer. Subsequent `cap.read()` calls overwrite the cv2 internal buffer, which is not the same memory as the `.copy()` numpy array. The `.clone()` requirement from the locked decisions is satisfied automatically by the `bgr[..., ::-1].copy()` + float division chain.

**Note:** If a future optimization uses `np.ascontiguousarray` instead of `.copy()`, the `.clone()` requirement must be re-evaluated.

```python
# Verified pattern — handles negative-stride and device requirements
def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = bgr[..., ::-1].copy()              # BGR->RGB, contiguous
    t = torch.from_numpy(rgb)                # (H, W, 3) uint8, shares numpy memory
    return t.permute(2, 0, 1).float() / 255.0  # (C, H, W) float32 [0,1], new tensor
```

### Recommended Project Structure

```
src/aquacore/
└── io/
    ├── __init__.py     # from .frameset import FrameSet; from .video import VideoSet;
    │                   # from .images import ImageSet, create_frameset
    ├── frameset.py     # FrameSet Protocol only
    ├── video.py        # VideoSet class only
    └── images.py       # ImageSet class + create_frameset factory function
```

**Placement of `create_frameset`:** Put it in `images.py` since it must import both `ImageSet` and `VideoSet`. Alternatively, it can be in `video.py`. Putting it in `__init__.py` creates circular imports. Either `images.py` or `video.py` works — `images.py` is preferred since `ImageSet` is the primary reference implementation.

### Anti-Patterns to Avoid

- **Returning NumPy arrays:** `__getitem__` and `__iter__` must return PyTorch tensors. The AquaMVS reference returns NumPy — diverge here explicitly.
- **Returning `(H, W, C)` layout:** Must permute to `(C, H, W)` before returning. Conv layers expect channel-first.
- **Not calling `.copy()` after negative-stride slice:** `bgr[..., ::-1]` creates a view with negative stride. `torch.from_numpy` on a negative-stride array raises `ValueError: some of the strides of a given numpy array are negative`. Always call `.copy()` first.
- **Forgetting context manager on VideoSet:** `cv2.VideoCapture` holds OS file handles. If not released, subsequent runs fail to open the same files on some platforms. Always use `with VideoSet(...) as vs:`.
- **Using `cap.read()` inside `__getitem__`:** `cap.read()` advances the internal frame pointer. In `__getitem__`, always call `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` before `cap.read()`. In `__iter__`, do NOT call `cap.set()` per frame — just call `cap.read()` sequentially.
- **Mixing seek and sequential iteration:** After `__getitem__` (which seeks), calling `__iter__` without resetting to frame 0 will produce frames starting from the wrong position. The `__iter__` implementation must call `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` at the start.
- **Not using `warnings.warn` for init-time mismatch:** Frame count mismatch is detected at construction time. Use `warnings.warn` (not `logging.warning`) for construction-time data quality issues per the project's established pattern from Phase 3.
- **Using Protocol inheritance instead of structural typing:** `ImageSet` and `VideoSet` must NOT inherit from `FrameSet`. They satisfy the Protocol structurally. Inheriting from a Protocol with `@runtime_checkable` works but defeats the purpose of structural typing and couples the concrete classes to the abstract type.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video file reading and seeking | Custom file parser | `cv2.VideoCapture` | Handles codec differences, keyframe seeking, platform-specific decoders |
| Image file reading | Custom PNG/JPEG decoder | `cv2.imread` | Handles all formats, color spaces, bit depths with single call |
| File type detection from extension | Custom extension dict | AquaMVS `detect_input_type` pattern with `_VIDEO_EXTENSIONS` set | Already proven in AquaMVS; edge cases (mixed types, nonexistent paths) handled |
| Protocol definition | Abstract base class with `abstractmethod` | `typing.Protocol` with `@runtime_checkable` | Structural typing — concrete classes don't need to import or inherit Protocol |
| BGR→RGB conversion | Custom channel swap loop | `array[..., ::-1].copy()` | Single numpy operation; `.copy()` is required for torch compatibility |
| Frame synchronization | Custom timestamp matching | Sorted filename matching (ImageSet) / sequential read (VideoSet) | Domain assumption: cameras are hardware-synchronized; no timestamp logic needed |

**Key insight:** The synchronization problem is solved at the hardware/filesystem level (synchronized capture → same filename / same frame number). The I/O layer merely enforces this alignment; it does not perform timestamp matching.

---

## Common Pitfalls

### Pitfall 1: Negative-Stride Array Passed to torch.from_numpy

**What goes wrong:** `torch.from_numpy(bgr[..., ::-1])` raises `ValueError: some of the strides of a given numpy array are negative`. Alternatively, this may raise a runtime error or silently produce wrong data depending on torch version.

**Why it happens:** `bgr[..., ::-1]` is a numpy view with a stride of -1 along the channel axis. PyTorch cannot wrap arrays with negative strides.

**How to avoid:** Always call `.copy()` after the channel-reversal slice: `rgb = bgr[..., ::-1].copy()`. This materializes a new contiguous C-order array.

**Warning signs:** Works in some torch versions, fails in others; error message mentions "strides" or "contiguous".

### Pitfall 2: __iter__ Starts Mid-Sequence After __getitem__ Calls

**What goes wrong:** After calling `vs[5]` (which seeks to frame 5), calling `for idx, frames in vs:` starts from frame 6 instead of frame 0, producing a truncated iteration with frames 6..N-1 yielded for indices 6..N-1. Earlier frames appear to be missing.

**Why it happens:** `cv2.VideoCapture` maintains an internal read pointer. `cap.set(CAP_PROP_POS_FRAMES, 5)` followed by `cap.read()` advances the pointer to 6. Subsequent `cap.read()` calls continue from wherever the pointer is.

**How to avoid:** The `__iter__` method must reset every capture to frame 0 at the start: `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` for all cameras before the loop begins.

**Warning signs:** Frame indices yielded by `__iter__` start at a non-zero value; first N frames missing from processing pipelines that mix `__getitem__` and `__iter__`.

### Pitfall 3: cv2.VideoCapture Frame Count is Inaccurate for Some Codecs

**What goes wrong:** `cap.get(cv2.CAP_PROP_FRAME_COUNT)` returns an incorrect value (off by 1 or 2, or returns 0 for some containers). Consumers discover the mismatch when the last `cap.read()` call returns `ok=False` before reaching the claimed frame count.

**Why it happens:** Some codecs (MKV, MOV) don't store frame count metadata in the header. OpenCV estimates the count from duration × FPS, which can be wrong. This is a known OpenCV limitation.

**How to avoid:** The `__iter__` implementation should stop on `ok=False` rather than strictly iterating `range(self._frame_count)`. Use `self._frame_count` as the upper bound but break if any camera returns `ok=False`. Alternatively, treat the frame count as a best-effort hint and let iteration drive termination.

**Decision (per locked context):** `__iter__` yields `(frame_idx, dict)`. If a camera fails at frame idx, it is omitted from the dict (warn + omit pattern). The loop should continue to the next frame. Break only when ALL cameras fail simultaneously — that signals end-of-video.

**Warning signs:** Iteration stops 1-2 frames early; last frame in a pipeline is always missing.

### Pitfall 4: VideoCapture Handle Leak if Constructor Raises Mid-Init

**What goes wrong:** If `_open_captures()` raises on the 3rd camera after successfully opening 2, the first 2 `VideoCapture` objects are leaked (never released). On some platforms this prevents reopening the same files until the GC collects them.

**Why it happens:** Exception during init skips `__exit__`, so context manager cleanup doesn't run.

**How to avoid:** In `_open_captures()`, if any camera raises, release all already-opened captures before re-raising the exception:

```python
try:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        # Release all previously opened captures
        for c in self._caps.values():
            c.release()
        raise ValueError(f"Cannot open video file: {path}")
    self._caps[cam_name] = cap
except Exception:
    for c in self._caps.values():
        c.release()
    raise
```

**Warning signs:** `ValueError` during construction, followed by `Cannot open video` errors on retry with the same files.

### Pitfall 5: FrameSet Protocol isinstance Check Fails for Non-Method Attributes

**What goes wrong:** `isinstance(vs, FrameSet)` returns `False` even though `VideoSet` implements all required methods.

**Why it happens:** `@runtime_checkable` Protocol `isinstance` checks only verify that the required methods exist as attributes. If there is a naming mismatch (e.g., `__iter__` is missing or misspelled), the check silently returns `False`.

**How to avoid:** Write a protocol compliance test that asserts `isinstance(ImageSet(...), FrameSet)` and `isinstance(VideoSet(...), FrameSet)` return `True`. This catches any method naming errors at test time.

**Warning signs:** Code written against `FrameSet` type annotation doesn't accept a `VideoSet` or `ImageSet` at runtime; mypy/basedpyright reports the concrete class as incompatible.

### Pitfall 6: Image Extension Globbing — Case Sensitivity and Extension Order

**What goes wrong:** On case-sensitive file systems (Linux), `cam_dir.glob("*.jpg")` finds `frame001.jpg` but not `frame001.JPG`. On Windows, both are found. Mixed-case extensions in a dataset cause silent frame count mismatches between cameras.

**Why it happens:** `pathlib.Path.glob` is case-sensitive on Linux/Mac, case-insensitive on Windows.

**How to avoid:** Include both cases in the extension list: `("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.tiff", "*.TIFF", "*.tif", "*.TIF")`. Alternatively, use `rglob` with a case-insensitive pattern. The AquaMVS reference only covers lowercase extensions — this is a known gap.

**Recommendation for this phase:** Include both cases explicitly. The project runs on Windows in development (CLAUDE.md env section), so this is primarily a future-proofing issue, but it's inexpensive to handle upfront.

### Pitfall 7: Frame Count from float vs int via CAP_PROP_FRAME_COUNT

**What goes wrong:** `cap.get(cv2.CAP_PROP_FRAME_COUNT)` returns a `float`. Using it directly in `range(cap.get(cv2.CAP_PROP_FRAME_COUNT))` raises `TypeError: 'float' object cannot be interpreted as an integer`.

**Why it happens:** All `cv2.VideoCapture.get()` calls return `float`, even for integer-valued properties.

**How to avoid:** Always cast: `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`.

**Warning signs:** `TypeError` on VideoSet construction; easy to miss because tests with small frame counts (created programmatically) may not hit the code path.

---

## Code Examples

Verified patterns from AquaMVS source and project-established conventions:

### BGR→RGB Tensor Conversion (Core Utility)

```python
# Source: numpy negative-stride behavior (documented) + torch.from_numpy requirements
import numpy as np
import torch

def _bgr_to_chw_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) uint8 BGR ndarray to (C, H, W) float32 [0, 1] tensor."""
    rgb = bgr[..., ::-1].copy()              # BGR->RGB; .copy() required for torch compat
    t = torch.from_numpy(rgb)                # (H, W, 3) uint8
    return t.permute(2, 0, 1).float() / 255.0  # (C, H, W) float32 [0, 1]
```

### FrameSet Protocol (Complete)

```python
# Source: Phase 2 ProjectionModel pattern (projection/protocol.py) — same structure
from typing import Iterator, Protocol, runtime_checkable
import torch

@runtime_checkable
class FrameSet(Protocol):
    """Protocol for synchronized multi-camera frame access."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]: ...
    def __enter__(self) -> "FrameSet": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

### create_frameset Factory

```python
# Source: AquaMVS/src/aquamvs/io.py detect_input_type() (adapted — returns instance)
from pathlib import Path
from .images import ImageSet
from .video import VideoSet

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}

def create_frameset(camera_map: dict[str, str | Path]) -> "ImageSet | VideoSet":
    if not camera_map:
        raise ValueError("camera_map must not be empty")
    paths = [Path(p) for p in camera_map.values()]
    is_dir = [p.is_dir() for p in paths]
    is_file = [p.is_file() for p in paths]
    if all(is_dir):
        return ImageSet(camera_map)
    if all(is_file):
        return VideoSet(camera_map)
    # Non-existent paths: infer from extension (for tests/mocks)
    if not any(is_dir) and not any(is_file):
        has_video_ext = [p.suffix.lower() in _VIDEO_EXTENSIONS for p in paths]
        if all(has_video_ext):
            return VideoSet(camera_map)
        if not any(has_video_ext):
            return ImageSet(camera_map)
    raise ValueError("camera_map contains mixed types (directories and files).")
```

### Test Pattern: Synthetic Image Directory (No Real Files)

```python
# Source: test_undistortion.py fixture pattern + tmp_path pytest fixture
import pytest
import numpy as np
import cv2
from pathlib import Path

@pytest.fixture
def two_camera_image_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create two temporary camera directories with 5 synthetic PNG frames each."""
    cam_names = ["cam0", "cam1"]
    camera_map = {}
    for cam_name in cam_names:
        cam_dir = tmp_path / cam_name
        cam_dir.mkdir()
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0] = i * 50  # Distinctive per-frame blue channel
            cv2.imwrite(str(cam_dir / f"frame_{i:04d}.png"), img)
        camera_map[cam_name] = cam_dir
    return camera_map
```

### Test Pattern: Protocol Compliance

```python
# Source: Phase 2 protocol compliance test pattern (test_projection/test_protocol.py)
from aquacore.io.frameset import FrameSet
from aquacore.io.images import ImageSet

def test_imageset_satisfies_frameset_protocol(two_camera_image_dirs):
    with ImageSet(two_camera_image_dirs) as img_set:
        assert isinstance(img_set, FrameSet), (
            "ImageSet must satisfy FrameSet protocol for isinstance() to return True"
        )
```

### Test Pattern: Tensor Format Verification

```python
# Source: test_undistortion.py shape/dtype pattern
import torch
from aquacore.io.images import ImageSet

def test_imageset_tensor_format(two_camera_image_dirs):
    with ImageSet(two_camera_image_dirs) as img_set:
        frames = img_set[0]
        for cam_name, tensor in frames.items():
            assert tensor.ndim == 3, f"{cam_name}: expected 3D tensor, got {tensor.ndim}D"
            assert tensor.shape[0] == 3, f"{cam_name}: expected C=3, got shape {tensor.shape}"
            assert tensor.dtype == torch.float32, f"{cam_name}: expected float32"
            assert tensor.min() >= 0.0 and tensor.max() <= 1.0, (
                f"{cam_name}: values out of [0, 1] range"
            )
```

### io/__init__.py Public API

```python
# Source: projection/__init__.py pattern (exports all public names)
"""Synchronized multi-camera frame I/O."""

from .frameset import FrameSet
from .images import ImageSet, create_frameset
from .video import VideoSet

__all__ = ["FrameSet", "ImageSet", "VideoSet", "create_frameset"]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| AquaMVS: `ImageDirectorySet.read_frame()` returns `dict[str, np.ndarray]` (H,W,3) uint8 | AquaCore: `ImageSet.__getitem__()` returns `dict[str, Tensor]` (C,H,W) float32 [0,1] | Phase 4 (this phase) | AquaCore frames are ML-ready without further conversion |
| AquaMVS: `iterate_frames(start, stop, step)` method | AquaCore: standard `__iter__` yielding `(idx, dict)` tuples | Phase 4 (this phase) | Enables `for idx, frames in image_set:` syntax; compatible with Python iteration protocol |
| AquaMVS: no VideoSet implementation | AquaCore: VideoSet with cv2.VideoCapture | Phase 4 (this phase) | AquaPose sequential video processing supported |
| AquaMVS: no FrameSet protocol | AquaCore: `FrameSet` runtime_checkable Protocol | Phase 4 (this phase) | Code written against FrameSet type-checks with both concrete classes |
| AquaMVS: `detect_input_type()` returns string "images"/"video" | AquaCore: `create_frameset()` returns concrete instance | Phase 4 (this phase) | Factory returns ready-to-use object; caller doesn't dispatch on string |

**Deprecated/outdated:**
- `read_frame()` / `iterate_frames()` names: AquaMVS API. AquaCore uses `__getitem__` / `__iter__`. Consumers porting from AquaMVS must update call sites.
- NumPy image return type: AquaMVS returns `np.ndarray`. AquaCore returns `torch.Tensor`. Consumers using `images["cam0"]` will get a different type.

---

## Claude's Discretion Recommendations

### Float32 Value Range: [0, 1]

**Recommendation: `[0, 1]`**

Rationale: The primary consumers (AquaPose for neural network inference, AquaMVS for dense matching) both follow torchvision convention where images are normalized to `[0, 1]`. The conversion is `/ 255.0` — a single scalar division that adds negligible cost. The `[0, 255]` range would require consumers to normalize before passing to any standard PyTorch model, creating friction. There is no consumer in the Aqua ecosystem that requires the `[0, 255]` range.

### Image Extensions: Both Cases

**Recommendation:** Include `*.png, *.PNG, *.jpg, *.JPG, *.jpeg, *.JPEG, *.tiff, *.TIFF, *.tif, *.TIF, *.bmp, *.BMP`

Rationale: Case-sensitivity on Linux would silently drop uppercase-extension files. On the current dev platform (Windows) this is not an issue, but the code should be portable. The AquaMVS reference uses only lowercase — this is a known gap worth closing.

### Logging Verbosity

**Recommendation:** `logging.info` for successful init (frame count, camera count); `warnings.warn` (stacklevel=2) for data quality issues at init (frame count mismatch); `warnings.warn` (stacklevel=3) for per-frame read failures (inside `_read_frame_dict` / `_bgr_to_tensor` helpers); no `logging.debug` unless a test explicitly enables it.

Rationale: Matches the Phase 3 pattern exactly. `warnings` for data quality; `logging` for operational info.

### VideoSet Sequential Buffering Strategy

**Recommendation:** No explicit buffering. Use cv2.VideoCapture's built-in sequential read (`cap.read()`) directly. Do not implement prefetch threads, ring buffers, or async read. The primary use case (AquaPose frame-by-frame processing) processes one frame at a time and does not benefit from prefetch. Complexity cost outweighs any benefit.

---

## Open Questions

1. **Should `__iter__` break when all cameras fail, or always run `range(self._frame_count)` iterations?**
   - What we know: `cap.get(CAP_PROP_FRAME_COUNT)` can be off by 1-2 frames for some codecs. If `_frame_count` is overestimated, the last iteration(s) will produce empty dicts.
   - What's unclear: Should an empty dict (all cameras failed) be yielded or silently skipped?
   - Recommendation: Yield empty dicts and let the consumer handle them. This preserves frame index semantics — `idx` always matches the actual position, even if no cameras produced a frame. Document this behavior.

2. **Should `io/__init__.py` re-export `create_frameset` or should the top-level `aquacore/__init__.py` be the only public entry point?**
   - What we know: The `projection/__init__.py` re-exports `project_multi` and `back_project_multi` (helper functions) in addition to the protocol and model. Consumers import from `aquacore` directly.
   - What's unclear: Whether `from aquacore.io import ImageSet` should work in addition to `from aquacore import ImageSet`.
   - Recommendation: Export from both. `io/__init__.py` exports the full subpackage surface; `aquacore/__init__.py` re-exports the same names for top-level access. This matches the `projection/` subpackage pattern.

3. **`__iter__` type annotation: `Iterator` vs `Generator`**
   - What we know: The method uses `yield`, making it a generator function. Its return type could be annotated as `Iterator[tuple[int, dict[str, Tensor]]]` or `Generator[tuple[int, dict[str, Tensor]], None, None]`.
   - Recommendation: Use `Iterator[tuple[int, dict[str, Tensor]]]` in the Protocol definition (broader type, any iterator satisfies it) and `Generator[...]` is not needed in the concrete class — `Iterator` is sufficient for the Protocol and for type-checkers. The `ProjectionModel` protocol uses `tuple[Tensor, Tensor]` return types for consistency — follow the same simplicity.

---

## Sources

### Primary (HIGH confidence — direct source code inspection on this machine)

- `C:/Users/tucke/PycharmProjects/AquaMVS/src/aquamvs/io.py` — Complete reference: `ImageDirectorySet` class, `detect_input_type()` function, image extension patterns, sorted filename matching, cv2.imread error handling, context manager pattern
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/projection/protocol.py` — ProjectionModel Protocol pattern: `@runtime_checkable`, method signatures, no inheritance requirement
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/io/frameset.py` — Current stub (empty); confirms file location and module docstring convention
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/io/video.py` — Current stub (empty)
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/io/images.py` — Current stub (empty)
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/io/__init__.py` — Current stub with empty `__all__`
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/__init__.py` — Current top-level API; will need FrameSet, VideoSet, ImageSet, create_frameset additions
- `C:/Users/tucke/PycharmProjects/AquaCore/src/aquacore/undistortion.py` — Established OpenCV boundary pattern: `.detach().cpu().numpy()` → cv2 operation → `torch.from_numpy().to(device)`
- `C:/Users/tucke/PycharmProjects/AquaCore/tests/conftest.py` — Device fixture pattern
- `C:/Users/tucke/PycharmProjects/AquaCore/tests/unit/test_undistortion.py` — Test structure reference: fixtures, shape/dtype assertions, synthetic data
- `C:/Users/tucke/PycharmProjects/AquaCore/pyproject.toml` — Dependencies (opencv-python>=4.8, numpy>=1.24, torch via hatch env); Python 3.11+; no new deps needed

### Secondary (MEDIUM confidence)

- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/research/aquamvs-map.md` — Pre-mapped AquaMVS I/O patterns: ImageDirectorySet class summary, calibration pattern
- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/research/shared-patterns.md` — Cross-repo conventions: OpenCV boundary, device handling
- `C:/Users/tucke/PycharmProjects/AquaCore/.planning/phases/04-i-o-layer/04-CONTEXT.md` — Locked decisions and Claude's discretion areas

### Tertiary (LOW confidence — none needed; all claims sourced from code)

None. All findings are sourced from direct code inspection.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries in pyproject.toml; verified against AquaMVS reference
- Architecture: HIGH — FrameSet Protocol mirrors Phase 2 ProjectionModel exactly; ImageSet mirrors AquaMVS ImageDirectorySet with documented divergences; VideoSet is fresh but uses documented cv2.VideoCapture API
- Pitfalls: HIGH — all identified from direct AquaMVS source inspection (negative-stride numpy) or cv2.VideoCapture documented behavior (seek pointer, float frame count, codec accuracy) or project code review (context manager leak)
- Tensor conversion: HIGH — BGR→RGB→(C,H,W) float32 is a standard, documented pattern; `.copy()` requirement is documented in numpy and torch documentation

**Research date:** 2026-02-18
**Valid until:** 2026-08-18 (stable domain — cv2.VideoCapture API is stable across OpenCV 4.x; torch.from_numpy numpy-stride behavior is stable; Protocol structural typing is stable in Python 3.11+)
