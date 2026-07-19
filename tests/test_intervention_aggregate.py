"""Tests for artifact-derived intervention validation aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mountaincar_rl.analysis.aggregate import aggregate_intervention_runs


def _write_run(run_dir: Path, *, validation_seeds: list[int]) -> None:
    run_dir.mkdir()
    summary = {
        "condition": "dqn_replay_shaping_weak",
        "training_seed": 0,
        "first_training_environment_completion_step": 1200,
        "first_training_assignment_success_step": None,
        "first_validation_environment_completion_step": 5000,
        "first_validation_assignment_success_step": None,
        "best_validation": {
            "validation_after_episode": 25,
            "environment_steps": 5000,
            "environment_completion_rate": 0.5,
            "assignment_success_rate": 0.0,
            "mean_steps_among_completions": 150.0,
            "median_steps_among_completions": 150.0,
            "mean_episode_return": -175.0,
        },
    }
    diagnostics = {
        "absolute_shaping_to_environment_ratio": 0.02,
        "final_replay_state_coverage": {"coverage_fraction": 0.4},
        "episodes_remaining_near_valley": 2,
        "episode_count": 25,
        "evaluation_reward": "original MountainCar-v0 reward only",
    }
    config = {
        "evaluation": {"validation_episode_seeds": validation_seeds},
        "agent": {
            "warmup_steps": 1000,
            "epsilon": {"decay_steps": 20000},
        },
        "reward_shaping": {
            "enabled": True,
            "beta": 0.1,
            "eta": 0.5,
            "energy_scale": 0.9,
        },
        "diagnostics": {"shaping_dominance_ratio": 0.5},
    }
    for name, value in (
        ("summary.json", summary),
        ("diagnostics.json", diagnostics),
        ("config.snapshot.json", config),
    ):
        (run_dir / name).write_text(json.dumps(value), encoding="utf-8")


def test_aggregate_uses_saved_metrics_and_marks_single_seed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, validation_seeds=[1000, 1001])

    report = aggregate_intervention_runs([run_dir])
    row = report["conditions"][0]

    assert report["final_test_used"] is False
    assert report["training_seed_count"] == 1
    assert row["selected_validation_completion_rate"] == 0.5
    assert row["selected_validation_assignment_success_rate"] == 0.0
    assert row["rejected_for_numeric_reward_dominance"] is False


def test_aggregate_rejects_held_out_seed_leakage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, validation_seeds=[1000, 2000])

    with pytest.raises(ValueError, match="held-out"):
        aggregate_intervention_runs([run_dir])
