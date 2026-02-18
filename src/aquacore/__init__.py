"""Aquacore."""

from importlib.metadata import PackageNotFoundError, version

__all__: list[str] = []

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
