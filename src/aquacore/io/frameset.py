"""FrameSet protocol for synchronized multi-camera frame access."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FrameSet(Protocol):
    """Protocol for synchronized multi-camera frame access.

    Any class implementing ``__getitem__``, ``__len__``, ``__iter__``,
    ``__enter__``, and ``__exit__`` with the correct signatures satisfies this
    protocol structurally — no import of ``FrameSet`` is needed in the
    implementing class.

    The ``@runtime_checkable`` decorator enables both static type-checking
    (basedpyright structural subtyping) and runtime ``isinstance()`` checks.

    Frame format contract:
        ``__getitem__`` returns ``dict[str, torch.Tensor]`` where each tensor
        has shape ``(C, H, W)``, dtype ``float32``, and values in ``[0, 1]``.
        Camera names (string keys) identify each view.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return all camera frames for a given frame index.

        Args:
            idx: Zero-based frame index. Must be in ``[0, len(self))``.

        Returns:
            Mapping from camera name to ``(C, H, W)`` float32 tensor in
            ``[0, 1]``. Cameras that fail to read a frame at ``idx`` are
            omitted from the returned dict (warn + omit pattern).

        Raises:
            IndexError: If ``idx`` is negative or ``>= len(self)``.
        """
        ...

    def __len__(self) -> int:
        """Return the total number of frames available.

        For multi-camera sets with mismatched frame counts, the minimum
        count across all cameras is used.

        Returns:
            Total frame count.
        """
        ...

    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate frames sequentially, yielding ``(frame_idx, frames)`` tuples.

        Args:
            (no arguments — iterator protocol)

        Yields:
            Tuple of ``(frame_idx, dict[str, Tensor])`` for each frame index
            from ``0`` to ``len(self) - 1``. For ``VideoSet``, this is
            frame-exact (sequential read). For ``ImageSet``, all access is
            exact (file-based).
        """
        ...

    def __enter__(self) -> "FrameSet":
        """Enter the context manager, returning self.

        Returns:
            The ``FrameSet`` instance.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context manager, releasing any held resources.

        For ``ImageSet``, this is a no-op (no resources to release).
        For ``VideoSet``, this releases all ``cv2.VideoCapture`` handles.

        Args:
            exc_type: Exception type, or ``None`` if no exception occurred.
            exc_val: Exception value, or ``None`` if no exception occurred.
            exc_tb: Exception traceback, or ``None`` if no exception occurred.
        """
        ...
