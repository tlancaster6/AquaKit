---
phase: 04-i-o-layer
plan: 02
subsystem: io
tags: [opencv, torch, protocol, structural-typing, video-io, frameset, factory]

# Dependency graph
requires:
  - phase: 04-01
    provides: FrameSet runtime_checkable Protocol and ImageSet implementation

provides:
  - VideoSet class satisfying FrameSet structurally (seek-based + sequential iteration)
  - create_frameset factory auto-detecting image dirs vs video files
  - aquacore.io public API (FrameSet, ImageSet, VideoSet, create_frameset)
  - aquacore top-level public API (all 4 I/O names re-exported)
  - 19 new tests: 13 VideoSet + 6 create_frameset factory

affects:
  - downstream consumers (AquaMVS, AquaPose) using VideoSet for synchronized video I/O
  - any code importing from aquacore top-level (4 new names now available)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - VideoSet seek-based __getitem__ via CAP_PROP_POS_FRAMES + cap.read()
    - VideoSet sequential __iter__: resets all captures to frame 0 before looping
    - Mid-init cleanup: track opened captures in list, release on exception before re-raise
    - create_frameset: filesystem existence check first, then extension inference for mocks
    - mp4v/MJPG fallback pattern in test fixtures for cross-platform video creation

key-files:
  created:
    - src/aquacore/io/video.py
    - tests/unit/test_io/conftest.py
    - tests/unit/test_io/test_videoset.py
    - tests/unit/test_io/test_factory.py
  modified:
    - src/aquacore/io/images.py
    - src/aquacore/io/__init__.py
    - src/aquacore/__init__.py

key-decisions:
  - "VideoSet does NOT inherit from FrameSet: structural typing only (same as ImageSet)"
  - "VideoSet __iter__ resets all captures to frame 0 at start: guarantees frame-exact sequential read regardless of prior seek state"
  - "Mid-init cleanup: opened captures tracked in list, released on any exception before re-raise"
  - "create_frameset uses filesystem existence check first, then extension inference for nonexistent (mock/test) paths"
  - "VideoSet import placed at top of images.py: no circular import risk (video.py has no images.py dependency)"

patterns-established:
  - "VideoSet: warn + omit pattern for per-frame read failures (consistent with ImageSet)"
  - "cap.get(cv2.CAP_PROP_FRAME_COUNT) cast to int: avoids float frame count issues"
  - "Test fixture: mp4v with .mp4 fallback to MJPG with .avi for cross-platform codec availability"
  - "Factory pattern: is_dir() / is_file() first, then suffix-based inference for non-existent paths"

# Metrics
duration: 6min
completed: 2026-02-19
---

# Phase 4 Plan 02: I/O Layer (VideoSet + Factory) Summary

**VideoSet with seek-based random access, frame-exact sequential iteration with auto-reset, mid-init handle cleanup, and create_frameset factory that auto-detects image directories vs video files — completes the full aquacore.io public API with top-level re-exports**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-02-19T00:22:44Z
- **Completed:** 2026-02-19T00:28:58Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Implemented `VideoSet` with `cv2.VideoCapture`-backed seek (`__getitem__`) and sequential iteration (`__iter__` resets all captures to frame 0), context manager releasing all handles on exit, and mid-init cleanup that releases already-opened captures if the constructor fails partway through
- Added `create_frameset` factory to `images.py` that auto-detects input type: existing directories → `ImageSet`, existing files → `VideoSet`, nonexistent paths with video extensions → `VideoSet`, otherwise → `ImageSet`, mixed types → `ValueError`
- Wired all 4 public names (`FrameSet`, `ImageSet`, `VideoSet`, `create_frameset`) into both `aquacore.io.__all__` and the top-level `aquacore.__all__` in alphabetical order
- Wrote 19 new tests: 13 `VideoSet` tests (construction, tensor format, iter indices, iter count, iter reset after seek, context manager cleanup, protocol compliance, missing file, directory path, index out of range, frame count mismatch warning, mid-init cleanup) and 6 `create_frameset` tests (image dirs, video files, empty map, mixed types, nonexistent video extension, nonexistent dir extension)

## Task Commits

Each task was committed atomically:

1. **Task 1: VideoSet implementation, create_frameset factory, and __init__.py exports** - `80a452e` (feat)
2. **Task 2: VideoSet and create_frameset tests** - `77689c7` (feat)

## Files Created/Modified

- `src/aquacore/io/video.py` - VideoSet class: validation, seek-based __getitem__, sequential __iter__ with frame-0 reset, context manager with handle release, mid-init cleanup
- `src/aquacore/io/images.py` - Added `from .video import VideoSet`, `_VIDEO_EXTENSIONS` set, and `create_frameset` factory function at end of file
- `src/aquacore/io/__init__.py` - Updated to export all 4 public names: FrameSet, ImageSet, VideoSet, create_frameset
- `src/aquacore/__init__.py` - Added `from .io import FrameSet, ImageSet, VideoSet, create_frameset`; added 4 names to `__all__` in alphabetical order
- `tests/unit/test_io/conftest.py` - `two_camera_video_files` fixture with mp4v/MJPG fallback
- `tests/unit/test_io/test_videoset.py` - 13 VideoSet tests with synthetic video fixtures
- `tests/unit/test_io/test_factory.py` - 6 create_frameset factory tests

## Decisions Made

- `VideoSet` does NOT inherit from `FrameSet` — structural typing only, consistent with the `ImageSet`/`ProjectionModel` pattern established in earlier phases
- `VideoSet.__iter__` resets all captures to frame 0 at the start of every call — guarantees frame-exact sequential read regardless of any prior `__getitem__` seek operations, matching the "frame-exact" contract in the FrameSet docstring
- `VideoSet` import moved to top of `images.py` (not deferred at bottom) — no circular import risk since `video.py` does not import from `images.py`
- `create_frameset` uses filesystem existence checks first (`is_dir()`, `is_file()`), then falls back to suffix-based inference for nonexistent paths — enables test/mock use without requiring files on disk

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unused import and import ordering in test files**

- **Found during:** Task 2 verification (`hatch run check`)
- **Issue:** `test_videoset.py` had `import warnings` (unused) and both test files had I001 (un-sorted import blocks per ruff)
- **Fix:** Removed unused `warnings` import; ran `ruff check --fix` to sort import blocks
- **Files modified:** `tests/unit/test_io/test_videoset.py`, `tests/unit/test_io/test_factory.py`
- **Verification:** `hatch run check` passes with 0 errors
- **Committed in:** `77689c7` (Task 2 commit — ruff format pre-commit hook applied the fix before commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Minor style fix only. No scope creep.

## Issues Encountered

- Ruff pre-commit hook (`ruff format`) reformatted a test file after initial staging, causing the first commit attempt to fail. Re-staged the reformatted file and committed successfully. Standard ruff format behaviour on Windows (CRLF → LF conversion).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `aquacore.io` is complete with all 4 public names exported: `FrameSet`, `ImageSet`, `VideoSet`, `create_frameset`
- Top-level `aquacore` namespace now includes all I/O types alongside geometry math
- `isinstance(VideoSet(...), FrameSet)` and `isinstance(ImageSet(...), FrameSet)` both return True
- 226 tests pass across all phases; `hatch run check` passes with 0 errors/warnings
- Phase 4 is complete. Phase 5 (if any) can consume the full I/O API without modification.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `src/aquacore/io/video.py` | FOUND |
| `src/aquacore/io/images.py` (create_frameset) | FOUND |
| `src/aquacore/io/__init__.py` (4 exports) | FOUND |
| `src/aquacore/__init__.py` (IO imports) | FOUND |
| `tests/unit/test_io/conftest.py` | FOUND |
| `tests/unit/test_io/test_videoset.py` | FOUND |
| `tests/unit/test_io/test_factory.py` | FOUND |
| Commit `80a452e` | FOUND |
| Commit `77689c7` | FOUND |

---
*Phase: 04-i-o-layer*
*Completed: 2026-02-19*
