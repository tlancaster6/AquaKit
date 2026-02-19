"""Synchronized multi-camera frame I/O."""

from .frameset import FrameSet
from .images import ImageSet, create_frameset
from .video import VideoSet

__all__ = ["FrameSet", "ImageSet", "VideoSet", "create_frameset"]
