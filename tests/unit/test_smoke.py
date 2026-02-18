"""Smoke tests for package import."""

import aquacore


def test_version():
    assert aquacore.__version__
