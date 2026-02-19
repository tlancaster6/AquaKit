"""VideoSet implementation of the FrameSet protocol."""

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class VideoSet:
    """Synchronized frame access from per-camera video files.

    Opens one ``cv2.VideoCapture`` per camera at construction time. Provides
    both random-access (seek-based) and sequential (frame-exact) iteration.
    All ``cv2.VideoCapture`` handles are released via the context manager.

    Does not inherit from ``FrameSet`` — satisfies the protocol structurally
    (structural typing).

    Example::

        camera_map = {
            "cam0": Path("data/cam0.mp4"),
            "cam1": Path("data/cam1.mp4"),
        }
        with VideoSet(camera_map) as vs:
            for idx, frames in vs:
                tensor = frames["cam0"]  # (C, H, W) float32 [0, 1]

    Args:
        camera_map: Mapping from camera name to video file path. All
            files must exist and be openable by ``cv2.VideoCapture``.

    Raises:
        ValueError: If any file does not exist, is a directory, or
            cannot be opened by ``cv2.VideoCapture``.
    """

    def __init__(self, camera_map: dict[str, str | Path]) -> None:
        self._paths: dict[str, Path] = {name: Path(p) for name, p in camera_map.items()}
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._frame_count: int = 0
        self._open_captures()

    def _open_captures(self) -> None:
        """Validate files and open one cv2.VideoCapture per camera.

        Opens captures in insertion order. If any capture fails to open,
        releases all already-opened captures before re-raising to prevent
        handle leaks.

        Raises:
            ValueError: If any file does not exist, is a directory, or
                ``cv2.VideoCapture.isOpened()`` returns False.
        """
        opened: list[str] = []
        try:
            for cam_name, path in self._paths.items():
                if not path.exists():
                    raise ValueError(f"Video file does not exist: {path}")
                if path.is_dir():
                    raise ValueError(f"Video path is a directory, not a file: {path}")

                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise ValueError(
                        f"cv2.VideoCapture failed to open video for camera "
                        f"'{cam_name}': {path}"
                    )
                self._caps[cam_name] = cap
                opened.append(cam_name)
        except Exception:
            # Release all already-opened captures before propagating exception.
            for name in opened:
                if name in self._caps:
                    self._caps[name].release()
                    del self._caps[name]
            raise

        # Determine frame count: use minimum across cameras, warn on mismatch.
        counts: dict[str, int] = {
            name: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for name, cap in self._caps.items()
        }
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            min_count = min(unique_counts)
            warnings.warn(
                f"Frame counts differ across cameras: {counts}. "
                f"Using minimum frame count: {min_count}.",
                UserWarning,
                stacklevel=2,
            )
            self._frame_count = min_count
        else:
            self._frame_count = next(iter(unique_counts)) if unique_counts else 0

        logger.info(
            "VideoSet: %d frames, %d cameras",
            self._frame_count,
            len(self._caps),
        )

    def _bgr_to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR (H, W, 3) uint8 array to a (C, H, W) float32 tensor.

        Args:
            bgr: BGR image array of shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns:
            RGB tensor of shape ``(3, H, W)`` and dtype ``float32`` with
            values in ``[0, 1]``.
        """
        # .copy() is required: bgr[..., ::-1] has negative stride which
        # torch.from_numpy cannot accept.
        rgb = bgr[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return all camera frames for the given index (seek-based access).

        Seeks each capture to frame ``idx`` using ``CAP_PROP_POS_FRAMES``,
        then reads one frame. Cameras that fail to read are warned and omitted.

        Args:
            idx: Zero-based frame index.

        Returns:
            Mapping from camera name to ``(C, H, W)`` float32 tensor in
            ``[0, 1]``. Cameras that fail to read at ``idx`` are omitted.

        Raises:
            IndexError: If ``idx`` is negative or ``>= len(self)``.
        """
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")

        result: dict[str, torch.Tensor] = {}
        for cam_name, cap in self._caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                warnings.warn(
                    f"Failed to read frame {idx} from camera '{cam_name}'",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            result[cam_name] = self._bgr_to_tensor(frame)
        return result

    def __len__(self) -> int:
        """Return total frame count (minimum across all cameras).

        Returns:
            Total number of frames available.
        """
        return self._frame_count

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate all frames sequentially (frame-exact sequential read).

        Resets all captures to frame 0 before starting iteration to ensure
        consistent behaviour regardless of prior seek operations.

        Yields:
            Tuple of ``(frame_idx, dict[str, Tensor])`` for each frame from
            ``0`` to ``len(self) - 1``. Cameras that fail to read a frame
            are warned and omitted from that frame's dict.
        """
        # Reset all captures to frame 0 before sequential iteration.
        for cap in self._caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in range(self._frame_count):
            result: dict[str, torch.Tensor] = {}
            for cam_name, cap in self._caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    warnings.warn(
                        f"Failed to read frame {frame_idx} from camera '{cam_name}' "
                        f"during sequential iteration",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                result[cam_name] = self._bgr_to_tensor(frame)
            yield frame_idx, result

    def __enter__(self) -> "VideoSet":
        """Enter context manager, returning self.

        Returns:
            This ``VideoSet`` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, releasing all cv2.VideoCapture handles.

        Args:
            exc_type: Exception type, or ``None`` if no exception occurred.
            exc_val: Exception value, or ``None`` if no exception occurred.
            exc_tb: Exception traceback, or ``None`` if no exception occurred.
        """
        for cap in self._caps.values():
            cap.release()
        self._caps = {}
