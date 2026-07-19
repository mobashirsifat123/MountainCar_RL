"""Tests for deterministic process and Gymnasium seeding."""

import random

import numpy as np
import pytest
import torch

from mountaincar_rl.utils.seeding import seed_everything


def _seeded_draws(seed: int) -> tuple[float, float, float, np.ndarray]:
    """Collect one draw from every RNG controlled by ``seed_everything``."""
    generator = seed_everything(seed)
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(()).item()),
        generator.normal(size=4),
    )


def test_seed_everything_repeats_python_numpy_and_torch_draws() -> None:
    """Reapplying a seed should reproduce independent process-level RNG draws."""
    first = _seeded_draws(2025)
    second = _seeded_draws(2025)

    assert first[:3] == pytest.approx(second[:3])
    np.testing.assert_array_equal(first[3], second[3])


def test_seed_everything_returns_numpy_generator() -> None:
    """Algorithms should receive a local RNG rather than relying only on globals."""
    generator = seed_everything(17)

    assert isinstance(generator, np.random.Generator)


def test_seed_everything_rejects_negative_seed() -> None:
    """Negative seeds are invalid for NumPy's deterministic seed protocol."""
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)
