"""Integration tests for reward decomposition and research diagnostics."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mountaincar_rl.experiments.train import _write_jsonl_atomic
from mountaincar_rl.experiments.training import run_learning_training
from mountaincar_rl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_shaped_training_logs_original_shaping_and_total_rewards(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_config(ROOT / "configs/smoke/dqn_shaped.yaml"))
    config["experiment"]["episodes"] = 1
    config["environment"]["max_episode_steps"] = 5
    config["agent"]["hidden_sizes"] = [8]
    config["agent"]["replay_capacity"] = 16
    config["agent"]["batch_size"] = 2
    config["agent"]["warmup_steps"] = 2
    config["agent"]["target_update_frequency"] = 2
    config["evaluation"]["validation_episode_seeds"] = [1000]
    output_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "checkpoints"

    summary = run_learning_training(
        config,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        overwrite=False,
        write_jsonl_atomic=_write_jsonl_atomic,
    )

    with (output_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episode = next(csv.DictReader(handle))
    transitions = [
        json.loads(line)
        for line in (output_dir / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with (output_dir / "validation_episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        validation = next(csv.DictReader(handle))
    diagnostics = json.loads(
        (output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )

    assert len(transitions) == 5
    for transition in transitions:
        assert transition["training_reward"] == pytest.approx(
            transition["original_reward"] + transition["shaping_reward"]
        )
        assert transition["shaped_reward"] == transition["training_reward"]
        assert transition["epsilon_before"] is not None
        assert transition["environment_steps"] >= 1
    assert float(episode["original_return"]) == pytest.approx(-5.0)
    assert float(episode["training_return"]) == pytest.approx(
        float(episode["original_return"]) + float(episode["shaping_return"])
    )
    assert float(episode["shaped_return"]) == float(episode["training_return"])
    assert episode["replay_coverage_cells"] != ""
    assert validation["original_return"] == validation["episode_return"]
    assert validation["shaping_return"] == ""
    assert validation["training_return"] == ""
    assert summary["evaluation_reward"] == "original MountainCar-v0 reward only"
    assert diagnostics["reward_shaping"]["beta"] == 0.5
    assert diagnostics["phase_space_trajectory_source"] == "trajectories.jsonl"
