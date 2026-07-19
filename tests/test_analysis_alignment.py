"""Regression checks for seed-level validation-curve alignment."""

from __future__ import annotations

import numpy as np
import pytest

from mountaincar_rl.analysis.make_all_figures import _aligned_validation_rates


def test_validation_alignment_retains_overshot_checkpoints() -> None:
    history = [
        {"environment_steps": 10_000, "assignment_success_rate": 0.0},
        {"environment_steps": 20_100, "assignment_success_rate": 0.2},
        {"environment_steps": 30_000, "assignment_success_rate": 0.4},
    ]
    aligned = _aligned_validation_rates(history, np.array([10_000, 20_000, 30_000]))
    assert aligned == pytest.approx([0.0, 0.19801980198, 0.4])


def test_validation_alignment_rejects_nonincreasing_steps() -> None:
    history = [
        {"environment_steps": 10_000, "assignment_success_rate": 0.0},
        {"environment_steps": 10_000, "assignment_success_rate": 0.1},
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        _aligned_validation_rates(history, np.array([10_000]))
