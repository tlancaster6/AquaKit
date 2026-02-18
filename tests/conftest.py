"""Shared pytest fixtures for all AquaCore tests."""

import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrized device fixture — yields CPU and (if available) CUDA.

    All geometry tests should accept this fixture to ensure device-agnostic
    correctness. Never call .cuda() directly in tests.
    """
    return torch.device(request.param)
