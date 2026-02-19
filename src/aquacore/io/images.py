"""ImageSet implementation of the FrameSet protocol."""

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import cv2
import torch

logger = logging.getLogger(__name__)

# Both-case extensions for cross-platform compatibility (case-sensitive filesystems).
_IMAGE_EXTENSIONS = (
    "*.png",
    "*.PNG",
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.tiff",
    "*.TIFF",
    "*.tif",
    "*.TIF",
    "*.bmp",
    "*.BMP",
)


class ImageSet:
    """Synchronized frame access from per-camera image directories.

    Validates directory existence, globs image files for each camera, enforces
    matching filenames across cameras, and builds the sorted frame index at
    construction time.

    Does not inherit from ``FrameSet`` — satisfies the protocol structurally
    (structural typing).

    Example::

        camera_map = {
            "cam0": Path("data/cam0"),
            "cam1": Path("data/cam1"),
        }
        with ImageSet(camera_map) as img_set:
            for idx, frames in img_set:
                tensor = frames["cam0"]  # (C, H, W) float32 [0, 1]

    Args:
        camera_map: Mapping from camera name to image directory path. All
            directories must exist, be non-empty, and contain images with
            matching filenames across cameras.

    Raises:
        ValueError: If any directory does not exist, is not a directory,
            contains no image files, or if filenames do not match across
            cameras.
    """

    def __init__(self, camera_map: dict[str, str | Path]) -> None:
        self._dirs: dict[str, Path] = {name: Path(p) for name, p in camera_map.items()}
        self._frame_files: dict[str, list[Path]] = {}
        self._frame_count: int = 0
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        """Validate directories and build the sorted frame index.

        Raises:
            ValueError: If any directory is missing, empty, or filenames
                differ across cameras.
        """
        for cam_name, cam_dir in self._dirs.items():
            if not cam_dir.exists():
                raise ValueError(f"Camera directory does not exist: {cam_dir}")
            if not cam_dir.is_dir():
                raise ValueError(f"Camera path is not a directory: {cam_dir}")

            # Collect files, deduplicating by name to handle case-insensitive
            # filesystems (e.g., Windows) where *.png and *.PNG match the same files.
            seen: dict[str, Path] = {}
            for ext in _IMAGE_EXTENSIONS:
                for f in cam_dir.glob(ext):
                    seen.setdefault(f.name, f)
            files = list(seen.values())

            if not files:
                raise ValueError(
                    f"No image files found in directory for camera '{cam_name}': {cam_dir}"
                )

            self._frame_files[cam_name] = sorted(files, key=lambda p: p.name)

        # Enforce matching filenames across all cameras (catches alignment issues).
        reference_cam = next(iter(self._frame_files))
        reference_names = [f.name for f in self._frame_files[reference_cam]]
        for cam_name, files in self._frame_files.items():
            names = [f.name for f in files]
            if names != reference_names:
                raise ValueError(
                    f"Image filenames do not match between camera '{reference_cam}' "
                    f"and camera '{cam_name}'. All camera directories must contain "
                    f"images with identical filenames."
                )

        # Warn on frame count mismatch and use minimum (not ValueError).
        counts = {name: len(files) for name, files in self._frame_files.items()}
        if len(set(counts.values())) != 1:
            min_count = min(counts.values())
            warnings.warn(
                f"Frame counts differ across cameras: {counts}. "
                f"Using minimum frame count: {min_count}.",
                UserWarning,
                stacklevel=2,
            )
            for cam_name in self._frame_files:
                self._frame_files[cam_name] = self._frame_files[cam_name][:min_count]

        self._frame_count = min(len(files) for files in self._frame_files.values())
        logger.info(
            "ImageSet: %d frames, %d cameras",
            self._frame_count,
            len(self._dirs),
        )

    def _read_frame_dict(self, idx: int) -> dict[str, torch.Tensor]:
        """Read one frame from all cameras, returning (C, H, W) float32 tensors.

        Cameras that fail to read are warned and omitted from the result.

        Args:
            idx: Zero-based frame index (caller ensures in-bounds).

        Returns:
            Mapping from camera name to ``(C, H, W)`` float32 tensor in
            ``[0, 1]``. Omits cameras whose images cannot be read.
        """
        result: dict[str, torch.Tensor] = {}
        for cam_name, files in self._frame_files.items():
            bgr = cv2.imread(str(files[idx]))
            if bgr is None:
                warnings.warn(
                    f"Failed to read image: {files[idx]} "
                    f"(camera '{cam_name}', frame {idx})",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            # BGR (H,W,3) uint8 -> RGB (H,W,3) uint8 -> (C,H,W) float32 [0,1].
            # .copy() is required: bgr[..., ::-1] has negative stride which
            # torch.from_numpy cannot accept.
            rgb = bgr[..., ::-1].copy()
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            result[cam_name] = tensor
        return result

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return all camera frames for the given index.

        Args:
            idx: Zero-based frame index.

        Returns:
            Mapping from camera name to ``(C, H, W)`` float32 tensor in
            ``[0, 1]``.

        Raises:
            IndexError: If ``idx`` is negative or ``>= len(self)``.
        """
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        return self._read_frame_dict(idx)

    def __len__(self) -> int:
        """Return total frame count (minimum across all cameras).

        Returns:
            Total number of frames available.
        """
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate all frames sequentially.

        Yields:
            Tuple of ``(frame_idx, dict[str, Tensor])`` for each frame from
            ``0`` to ``len(self) - 1``.
        """
        for idx in range(self._frame_count):
            yield idx, self._read_frame_dict(idx)

    def __enter__(self) -> "ImageSet":
        """Enter context manager, returning self.

        Returns:
            This ``ImageSet`` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager (no-op — no resources to release).

        Args:
            exc_type: Exception type, or ``None`` if no exception occurred.
            exc_val: Exception value, or ``None`` if no exception occurred.
            exc_tb: Exception traceback, or ``None`` if no exception occurred.
        """
        pass  # No resources to release for image directory access.
