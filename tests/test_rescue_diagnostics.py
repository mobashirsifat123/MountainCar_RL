"""Leakage guards for controlled rescue analysis."""

from pathlib import Path

import pytest

from mountaincar_rl.analysis.rescue_diagnostics import _assert_validation_only


def test_rescue_diagnostics_reject_final_test_artifacts() -> None:
    with pytest.raises(ValueError, match="Held-out artifact forbidden"):
        _assert_validation_only(
            Path("results/raw/full/dqn_replay_seed0/final_test/episodes.csv")
        )


def test_rescue_diagnostics_accept_training_and_validation_artifacts() -> None:
    _assert_validation_only(Path("results/raw/full/dqn_replay_seed0/episodes.csv"))
    _assert_validation_only(
        Path("results/raw/full/dqn_replay_seed0/validation_episodes.csv")
    )
