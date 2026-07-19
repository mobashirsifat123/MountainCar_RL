"""Integration tests for exact interaction budgets and durable run resumption."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import torch

from mountaincar_rl.experiments.train import _write_jsonl_atomic
from mountaincar_rl.experiments.training import _score, run_learning_training
from mountaincar_rl.utils.checkpoints import load_agent_checkpoint
from mountaincar_rl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _budget_config() -> dict[str, object]:
    config = deepcopy(load_config(ROOT / "configs/smoke/dqn_replay.yaml"))
    config["experiment"]["episodes"] = 10
    config["experiment"]["max_environment_steps"] = 12
    config["environment"]["max_episode_steps"] = 5
    config["agent"]["hidden_sizes"] = [8]
    config["agent"]["replay_capacity"] = 32
    config["agent"]["batch_size"] = 2
    config["agent"]["warmup_steps"] = 2
    config["agent"]["target_update_frequency"] = 2
    config["training"]["validation_interval_steps"] = 5
    config["training"]["snapshot_interval_episodes"] = 1
    config["evaluation"]["validation_episode_seeds"] = [1000]
    return config


def test_interrupted_metrics_resume_to_exact_budget_and_match_clean_run(
    tmp_path: Path,
) -> None:
    config = _budget_config()
    resumed_output = tmp_path / "resumed"
    resumed_checkpoints = tmp_path / "resumed-checkpoints"

    partial = run_learning_training(
        config,
        output_dir=resumed_output,
        checkpoint_dir=resumed_checkpoints,
        overwrite=False,
        write_jsonl_atomic=_write_jsonl_atomic,
        max_episodes_this_invocation=1,
    )

    assert partial["status"] == "interrupted"
    assert partial["total_environment_interactions"] == 5
    assert (resumed_checkpoints / "latest.pt").is_file()
    assert not (resumed_checkpoints / "final.pt").exists()
    with (resumed_output / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 1
    for line in (resumed_output / "trajectories.jsonl").read_text().splitlines():
        assert isinstance(json.loads(line), dict)

    completed = run_learning_training(
        config,
        output_dir=resumed_output,
        checkpoint_dir=resumed_checkpoints,
        overwrite=False,
        write_jsonl_atomic=_write_jsonl_atomic,
        resume=True,
    )
    clean_output = tmp_path / "clean"
    clean_checkpoints = tmp_path / "clean-checkpoints"
    clean = run_learning_training(
        config,
        output_dir=clean_output,
        checkpoint_dir=clean_checkpoints,
        overwrite=False,
        write_jsonl_atomic=_write_jsonl_atomic,
    )

    assert completed["status"] == clean["status"] == "complete"
    assert completed["total_environment_interactions"] == 12
    assert clean["total_environment_interactions"] == 12
    assert (resumed_checkpoints / "best.pt").is_file()
    assert (resumed_checkpoints / "latest.pt").is_file()
    assert (resumed_checkpoints / "final.pt").is_file()
    resumed = load_agent_checkpoint(resumed_checkpoints / "final.pt")
    uninterrupted = load_agent_checkpoint(clean_checkpoints / "final.pt")
    assert resumed["training_state"]["total_interactions"] == 12
    assert resumed["agent_state"]["environment_steps"] == 12
    for key, value in resumed["agent_state"]["online_network"].items():
        torch.testing.assert_close(
            value, uninterrupted["agent_state"]["online_network"][key]
        )


def test_validation_score_uses_required_lexicographic_priority() -> None:
    base = {
        "assignment_success_rate": 0.2,
        "environment_completion_rate": 0.8,
        "median_steps_among_completions": 120.0,
        "episode_length_std": 15.0,
    }
    better_median = {**base, "median_steps_among_completions": 110.0}
    better_stability = {**base, "episode_length_std": 10.0}
    higher_completion = {**base, "environment_completion_rate": 0.9}
    higher_assignment = {**base, "assignment_success_rate": 0.3}

    assert _score(higher_assignment) > _score(higher_completion) > _score(base)
    assert _score(better_median) > _score(better_stability) > _score(base)
