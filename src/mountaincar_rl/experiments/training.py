"""Learning-agent training with deterministic validation and shared checkpoints."""

from __future__ import annotations

import math
import shlex
import sys
import time
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mountaincar_rl.agents.dqn import DQNAgent
from mountaincar_rl.agents.sarsa_lambda import SarsaLambdaAgent
from mountaincar_rl.analysis.statistics import summarize_episodes
from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.environments.reward_shaping import build_reward_shaper
from mountaincar_rl.experiments.agent_factory import (
    build_agent,
    configured_agent_type,
    preprocessing_metadata,
    require_mapping,
)
from mountaincar_rl.experiments.evaluation import EvaluationEpisode, evaluate_agent
from mountaincar_rl.experiments.rollout import training_episode_seeds
from mountaincar_rl.utils.checkpoints import (
    config_fingerprint,
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from mountaincar_rl.utils.logging import write_csv, write_json
from mountaincar_rl.utils.metadata import collect_run_metadata
from mountaincar_rl.utils.seeding import seed_everything


TRAIN_EPISODE_FIELDNAMES = [
    "condition",
    "split",
    "training_seed",
    "episode",
    "environment_seed",
    "steps",
    "episode_return",
    "original_return",
    "shaping_return",
    "training_return",
    "shaped_return",
    "terminated",
    "truncated",
    "environment_completion",
    "assignment_success",
    "epsilon_start",
    "epsilon",
    "mean_update_metric",
    "environment_steps",
    "optimizer_steps",
    "minimum_position",
    "maximum_position",
    "position_span",
    "maximum_absolute_velocity",
    "normalized_action_entropy",
    "remained_near_valley",
    "replay_buffer_size",
    "replay_coverage_cells",
    "replay_coverage_total_cells",
    "replay_coverage_fraction",
    "mean_absolute_shaping_reward",
    "absolute_shaping_to_environment_ratio",
    "action_count_0",
    "action_count_1",
    "action_count_2",
    "mean_q_value",
    "mean_gradient_norm",
    "wall_clock_seconds",
    "device",
    "config_hash",
    "git_commit",
]

VALIDATION_EPISODE_FIELDNAMES = [
    "condition",
    "split",
    "training_seed",
    "episode",
    "environment_seed",
    "steps",
    "episode_return",
    "original_return",
    "shaping_return",
    "training_return",
    "shaped_return",
    "terminated",
    "truncated",
    "environment_completion",
    "assignment_success",
    "validation_after_episode",
    "validation_environment_steps",
    "checkpoint_type",
    "maximum_position",
    "maximum_absolute_velocity",
    "action_count_0",
    "action_count_1",
    "action_count_2",
    "wall_clock_seconds",
    "device",
    "config_hash",
    "git_commit",
]


def _counter(agent: Any, *names: str) -> int:
    for name in names:
        value = getattr(agent, name, None)
        if value is not None:
            return int(value)
    return 0


def _epsilon(agent: Any) -> float | None:
    for name in ("current_epsilon", "epsilon"):
        value = getattr(agent, name, None)
        if value is not None:
            if callable(value):
                value = value()
            return float(value)
    return None


def _episode_record(
    *,
    condition: str,
    training_seed: int,
    episode: int,
    environment_seed: int,
    steps: int,
    original_return: float,
    shaping_return: float,
    training_return: float,
    absolute_environment_reward: float,
    absolute_shaping_reward: float,
    terminated: bool,
    truncated: bool,
    final_observation: np.ndarray,
    goal_position: float,
    update_metrics: Sequence[float],
    epsilon_start: float | None,
    positions: Sequence[float],
    velocities: Sequence[float],
    actions: Sequence[int],
    remained_near_valley: bool,
    replay_coverage: Mapping[str, int | float] | None,
    q_values: Sequence[float],
    gradient_norms: Sequence[float],
    wall_clock_seconds: float,
    device: str,
    config_hash: str,
    git_commit: str | None,
    agent: Any,
) -> dict[str, Any]:
    completed = bool(float(final_observation[0]) >= goal_position)
    return {
        "condition": condition,
        "split": "train",
        "training_seed": training_seed,
        "episode": episode,
        "environment_seed": environment_seed,
        "steps": steps,
        "episode_return": float(original_return),
        "original_return": float(original_return),
        "shaping_return": float(shaping_return),
        "training_return": float(training_return),
        "shaped_return": float(training_return),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "environment_completion": completed,
        "assignment_success": bool(completed and steps <= 100),
        "epsilon_start": epsilon_start,
        "epsilon": _epsilon(agent),
        "mean_update_metric": (
            float(np.mean(update_metrics)) if update_metrics else None
        ),
        "environment_steps": _counter(agent, "environment_steps", "env_steps", "update_count"),
        "optimizer_steps": _counter(agent, "optimizer_steps", "optimization_steps"),
        "minimum_position": float(min(positions)),
        "maximum_position": float(max(positions)),
        "position_span": float(max(positions) - min(positions)),
        "maximum_absolute_velocity": float(max(abs(value) for value in velocities)),
        "normalized_action_entropy": _normalized_action_entropy(
            actions, int(getattr(agent, "action_count", 3))
        ),
        "remained_near_valley": bool(remained_near_valley),
        "replay_buffer_size": (
            None if replay_coverage is None else int(replay_coverage["buffer_size"])
        ),
        "replay_coverage_cells": (
            None if replay_coverage is None else int(replay_coverage["occupied_cells"])
        ),
        "replay_coverage_total_cells": (
            None if replay_coverage is None else int(replay_coverage["total_cells"])
        ),
        "replay_coverage_fraction": (
            None
            if replay_coverage is None
            else float(replay_coverage["coverage_fraction"])
        ),
        "mean_absolute_shaping_reward": float(
            absolute_shaping_reward / max(steps, 1)
        ),
        "absolute_shaping_to_environment_ratio": float(
            absolute_shaping_reward / max(absolute_environment_reward, 1e-12)
        ),
        "action_count_0": int(sum(action == 0 for action in actions)),
        "action_count_1": int(sum(action == 1 for action in actions)),
        "action_count_2": int(sum(action == 2 for action in actions)),
        "mean_q_value": float(np.mean(q_values)) if q_values else None,
        "mean_gradient_norm": (
            float(np.mean(gradient_norms)) if gradient_norms else None
        ),
        "wall_clock_seconds": float(wall_clock_seconds),
        "device": device,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }


def _normalized_action_entropy(actions: Sequence[int], action_count: int) -> float:
    """Return empirical action entropy divided by its categorical maximum."""

    if not actions or action_count <= 1:
        return 0.0
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=action_count)
    probabilities = counts[counts > 0].astype(np.float64) / len(actions)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / np.log(action_count))


def _validation_record(
    episode: EvaluationEpisode,
    *,
    condition: str,
    training_seed: int,
    episode_index: int,
    validation_after_episode: int,
    validation_environment_steps: int,
    checkpoint_type: str,
    wall_clock_seconds: float,
    device: str,
    config_hash: str,
    git_commit: str | None,
) -> dict[str, Any]:
    positions = episode.position_trajectory
    velocities = episode.velocity_trajectory
    actions = episode.action_trajectory
    return {
        "condition": condition,
        "split": "validation",
        "training_seed": training_seed,
        "episode": episode_index,
        "environment_seed": episode.environment_seed,
        "steps": episode.steps,
        "episode_return": episode.original_return,
        "original_return": episode.original_return,
        "shaping_return": None,
        "training_return": None,
        "shaped_return": episode.shaped_return,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "environment_completion": episode.environment_completion,
        "assignment_success": episode.assignment_success,
        "validation_after_episode": validation_after_episode,
        "validation_environment_steps": validation_environment_steps,
        "checkpoint_type": checkpoint_type,
        "maximum_position": max(positions) if positions else None,
        "maximum_absolute_velocity": (
            max(abs(value) for value in velocities) if velocities else None
        ),
        "action_count_0": sum(action == 0 for action in actions),
        "action_count_1": sum(action == 1 for action in actions),
        "action_count_2": sum(action == 2 for action in actions),
        "wall_clock_seconds": float(wall_clock_seconds),
        "device": device,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }


def _evaluation_trajectory_records(
    episode: EvaluationEpisode,
    *,
    condition: str,
    training_seed: int,
    episode_index: int,
    validation_after_episode: int,
    validation_environment_steps: int,
    config_hash: str,
    device: str,
    git_commit: str | None,
) -> list[dict[str, Any]]:
    records = episode.to_transition_records(
        condition=condition,
        split="validation",
        training_seed=training_seed,
        episode_index=episode_index,
    )
    for record in records:
        record["validation_after_episode"] = validation_after_episode
        record["validation_environment_steps"] = validation_environment_steps
        record["checkpoint_type"] = "periodic_validation"
        record["config_hash"] = config_hash
        record["device"] = device
        record["git_commit"] = git_commit
    return records


def _score(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return the preregistered lexicographic validation checkpoint score.

    Priority is assignment success, completion, lower median completed length,
    then lower standard deviation of episode lengths. The caller replaces a
    checkpoint only on strict improvement, making the earliest checkpoint the
    deterministic final tie-breaker.
    """

    median = summary["median_steps_among_completions"]
    return (
        float(summary["assignment_success_rate"]),
        float(summary["environment_completion_rate"]),
        -1.0e12 if median is None else -float(median),
        -float(summary["episode_length_std"]),
    )


_BOOLEAN_FIELDS = {
    "terminated",
    "truncated",
    "environment_completion",
    "assignment_success",
    "remained_near_valley",
}
_INTEGER_FIELDS = {
    "training_seed",
    "episode",
    "environment_seed",
    "steps",
    "environment_steps",
    "optimizer_steps",
    "replay_buffer_size",
    "replay_coverage_cells",
    "replay_coverage_total_cells",
    "action_count_0",
    "action_count_1",
    "action_count_2",
    "validation_after_episode",
    "validation_environment_steps",
}
_FLOAT_FIELDS = set(TRAIN_EPISODE_FIELDNAMES + VALIDATION_EPISODE_FIELDNAMES).difference(
    _BOOLEAN_FIELDS | _INTEGER_FIELDS | {
        "condition",
        "split",
        "device",
        "config_hash",
        "git_commit",
        "checkpoint_type",
    }
)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read one persisted metrics snapshot back into typed row mappings."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in _BOOLEAN_FIELDS:
                    if value not in {"True", "False"}:
                        raise ValueError(f"Invalid boolean {value!r} in {path}:{key}")
                    row[key] = value == "True"
                elif key in _INTEGER_FIELDS:
                    row[key] = int(value)
                elif key in _FLOAT_FIELDS:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row at {path}:{line_number} must be a mapping")
        rows.append(row)
    return rows


def _write_training_snapshot(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    agent_type: str,
    agent: Any,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    validation_trajectory_rows: Sequence[Mapping[str, Any]],
    validation_history: Sequence[Mapping[str, Any]],
    training_state: Mapping[str, Any],
    best_selection: Mapping[str, Any] | None,
    write_jsonl_atomic: Any,
) -> None:
    """Atomically refresh durable metrics and the resumable latest checkpoint."""

    write_csv(output_dir / "episodes.csv", episode_rows, TRAIN_EPISODE_FIELDNAMES)
    write_csv(
        output_dir / "validation_episodes.csv",
        validation_rows,
        VALIDATION_EPISODE_FIELDNAMES,
    )
    write_jsonl_atomic(output_dir / "trajectories.jsonl", transition_rows)
    write_jsonl_atomic(
        output_dir / "validation_trajectories.jsonl", validation_trajectory_rows
    )
    write_json(output_dir / "validation_history.json", validation_history)
    write_json(output_dir / "run_state.json", dict(training_state))
    save_agent_checkpoint(
        checkpoint_path,
        agent_type=agent_type,
        agent_state=agent.state_dict(),
        config=config,
        training_state=training_state,
        preprocessing=preprocessing_metadata(agent),
        selection=best_selection,
    )


def run_learning_training(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    overwrite: bool,
    write_jsonl_atomic: Any,
    resume: bool = False,
    max_episodes_this_invocation: int | None = None,
) -> dict[str, Any]:
    """Train SARSA or either DQN ablation and persist auditable artifacts."""

    experiment = require_mapping(config, "experiment")
    environment_config = require_mapping(config, "environment")
    training_config = require_mapping(config, "training")
    logging_config = require_mapping(config, "logging")
    reproducibility = require_mapping(config, "reproducibility")
    evaluation_config = require_mapping(config, "evaluation")
    agent_type = configured_agent_type(config)
    if agent_type not in {"sarsa_lambda", "dqn"}:
        raise ValueError("run_learning_training supports sarsa_lambda or dqn.")
    if environment_config.get("id") != "MountainCar-v0":
        raise ValueError("Only environment.id='MountainCar-v0' is supported.")

    training_seed = int(experiment["training_seed"])
    episode_count = int(experiment["episodes"])
    max_environment_steps_value = experiment.get("max_environment_steps")
    max_environment_steps = (
        None
        if max_environment_steps_value is None
        else int(max_environment_steps_value)
    )
    max_episode_steps = int(environment_config.get("max_episode_steps", 200))
    condition = str(experiment["condition"])
    validation_interval = int(training_config.get("validation_interval_episodes", 1))
    validation_interval_steps_value = training_config.get("validation_interval_steps")
    validation_interval_steps = (
        None
        if validation_interval_steps_value is None
        else int(validation_interval_steps_value)
    )
    snapshot_interval = int(
        training_config.get("snapshot_interval_episodes", validation_interval)
    )
    if episode_count < 1 or validation_interval < 1 or snapshot_interval < 1:
        raise ValueError("episode and interval settings must be positive.")
    if max_environment_steps is not None and max_environment_steps < 1:
        raise ValueError("experiment.max_environment_steps must be positive.")
    if validation_interval_steps is not None and validation_interval_steps < 1:
        raise ValueError("training.validation_interval_steps must be positive.")
    if max_episodes_this_invocation is not None and max_episodes_this_invocation < 1:
        raise ValueError("max_episodes_this_invocation must be positive.")
    validation_seeds = [
        int(seed) for seed in evaluation_config["validation_episode_seeds"]
    ]
    if not validation_seeds:
        raise ValueError("validation_episode_seeds must not be empty.")
    record_trajectories = bool(
        logging_config.get(
            "record_training_trajectories",
            logging_config.get("record_trajectories", True),
        )
    )
    record_validation_trajectories = bool(
        logging_config.get(
            "record_validation_trajectories",
            logging_config.get("record_trajectories", True),
        )
    )
    deterministic_torch = bool(reproducibility.get("deterministic_torch", True))
    device = str(reproducibility.get("device", "cpu"))
    diagnostics_config = config.get("diagnostics", {})
    if not isinstance(diagnostics_config, Mapping):
        raise ValueError("Configuration key 'diagnostics' must be a mapping.")
    coverage_bins_value = diagnostics_config.get("replay_coverage_bins", (20, 20))
    if not isinstance(coverage_bins_value, Sequence) or isinstance(
        coverage_bins_value, (str, bytes)
    ):
        raise ValueError("diagnostics.replay_coverage_bins must be a sequence.")
    coverage_bins = tuple(int(value) for value in coverage_bins_value)
    valley_center = float(
        diagnostics_config.get("valley_center_position", -np.pi / 6.0)
    )
    valley_radius = float(diagnostics_config.get("valley_position_radius", 0.20))
    valley_velocity_threshold = float(
        diagnostics_config.get("valley_velocity_threshold", 0.025)
    )
    shaping_dominance_ratio = float(
        diagnostics_config.get("shaping_dominance_ratio", 0.5)
    )
    if valley_radius <= 0.0 or valley_velocity_threshold <= 0.0:
        raise ValueError("valley diagnostic thresholds must be positive.")
    if shaping_dominance_ratio <= 0.0:
        raise ValueError("diagnostics.shaping_dominance_ratio must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    final_path = checkpoint_dir / "final.pt"
    last_path = checkpoint_dir / "last.pt"
    conflicts = [
        path for path in (best_path, latest_path, final_path, last_path) if path.exists()
    ]
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if resume and not latest_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {latest_path}")
    if conflicts and not overwrite and not resume:
        raise FileExistsError(
            "Checkpoint target(s) already exist: "
            + ", ".join(str(path) for path in conflicts)
        )

    seed_everything(training_seed, deterministic_torch=deterministic_torch)
    started = time.perf_counter()
    metadata = collect_run_metadata(dict(config), device=device)
    metadata["command"] = shlex.join(sys.argv)
    configuration_hash = config_fingerprint(config)
    git_commit = metadata["git"]["commit"]
    write_json(output_dir / "config.snapshot.json", dict(config))
    write_json(output_dir / "metadata.json", metadata)
    env = make_env(seed=training_seed, max_episode_steps=max_episode_steps)
    agent = build_agent(config, env)
    episode_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    validation_trajectory_rows: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_score: tuple[float, float, float, float] | None = None
    best_selection: dict[str, Any] | None = None
    train_seeds = training_episode_seeds(training_seed, episode_count)
    goal_position = float(env.unwrapped.goal_position)
    agent_gamma = float(getattr(agent, "gamma"))
    reward_config = config.get("reward_shaping")
    if reward_config is not None and not isinstance(reward_config, Mapping):
        raise ValueError("Configuration key 'reward_shaping' must be a mapping.")
    reward_shaper = build_reward_shaper(
        reward_config,
        gamma=agent_gamma,
        goal_position=goal_position,
    )
    total_interactions = 0
    total_absolute_environment_reward = 0.0
    total_absolute_shaping_reward = 0.0
    first_training_completion_step: int | None = None
    first_training_assignment_success_step: int | None = None
    first_validation_completion_step: int | None = None
    first_validation_assignment_success_step: int | None = None
    completed_episodes = 0
    next_validation_step = validation_interval_steps

    if resume:
        payload = load_agent_checkpoint(
            latest_path,
            map_location=device,
            expected_agent_type=agent_type,
            expected_config=config,
        )
        agent.load_state_dict(payload["agent_state"])
        saved_state = payload["training_state"]
        completed_episodes = int(saved_state["completed_episodes"])
        total_interactions = int(saved_state["total_interactions"])
        total_absolute_environment_reward = float(
            saved_state["total_absolute_environment_reward"]
        )
        total_absolute_shaping_reward = float(
            saved_state["total_absolute_shaping_reward"]
        )
        first_training_completion_step = saved_state.get(
            "first_training_completion_step"
        )
        first_training_assignment_success_step = saved_state.get(
            "first_training_assignment_success_step"
        )
        first_validation_completion_step = saved_state.get(
            "first_validation_completion_step"
        )
        first_validation_assignment_success_step = saved_state.get(
            "first_validation_assignment_success_step"
        )
        raw_score = saved_state.get("best_score")
        best_score = None if raw_score is None else tuple(float(x) for x in raw_score)
        best_selection = saved_state.get("best_selection")
        next_validation_step = saved_state.get("next_validation_step")
        episode_rows = _read_csv_rows(output_dir / "episodes.csv")
        transition_rows = _read_jsonl_rows(output_dir / "trajectories.jsonl")
        validation_rows = _read_csv_rows(output_dir / "validation_episodes.csv")
        validation_trajectory_rows = _read_jsonl_rows(
            output_dir / "validation_trajectories.jsonl"
        )
        history_path = output_dir / "validation_history.json"
        validation_history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
        episode_rows = [
            row for row in episode_rows if int(row["episode"]) < completed_episodes
        ]
        transition_rows = [
            row for row in transition_rows if int(row["episode"]) < completed_episodes
        ]
        validation_rows = [
            row
            for row in validation_rows
            if int(row["validation_after_episode"]) <= completed_episodes
        ]
        validation_trajectory_rows = [
            row
            for row in validation_trajectory_rows
            if int(row["validation_after_episode"]) <= completed_episodes
        ]
        validation_history = [
            row
            for row in validation_history
            if int(row["validation_after_episode"]) <= completed_episodes
        ]

    invocation_episodes = 0
    stopped_early = False
    try:
        for episode_index in range(completed_episodes, episode_count):
            if (
                max_environment_steps is not None
                and total_interactions >= max_environment_steps
            ):
                break
            if (
                max_episodes_this_invocation is not None
                and invocation_episodes >= max_episodes_this_invocation
            ):
                stopped_early = True
                break
            environment_seed = train_seeds[episode_index]
            observation, _ = env.reset(seed=environment_seed)
            env.action_space.seed(environment_seed)
            agent.end_episode()
            action = int(agent.select_action(np.asarray(observation), explore=True))
            terminated = False
            truncated = False
            step_count = 0
            original_return = 0.0
            shaping_return = 0.0
            training_return = 0.0
            absolute_environment_reward = 0.0
            absolute_shaping_reward = 0.0
            update_metrics: list[float] = []
            q_value_metrics: list[float] = []
            gradient_norm_metrics: list[float] = []
            episode_positions = [float(observation[0])]
            episode_velocities = [float(observation[1])]
            episode_actions: list[int] = []
            epsilon_at_episode_start = _epsilon(agent)

            while not (terminated or truncated):
                next_observation, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                total_interactions += 1
                if (
                    max_environment_steps is not None
                    and total_interactions >= max_environment_steps
                    and not (terminated or truncated)
                ):
                    truncated = True
                    info = dict(info)
                    info["interaction_budget_truncated"] = True
                epsilon_before = _epsilon(agent)
                selected_q_value = float(
                    agent.q_value(np.asarray(observation), action)
                    if isinstance(agent, SarsaLambdaAgent)
                    else agent.q_values(np.asarray(observation))[action]
                )
                q_value_metrics.append(selected_q_value)
                reward_components = reward_shaper.reward_components(
                    reward,
                    np.asarray(observation),
                    np.asarray(next_observation),
                    terminated=bool(terminated),
                )
                original_return += reward_components.original_reward
                shaping_return += reward_components.shaping_reward
                training_return += reward_components.training_reward
                absolute_environment_reward += abs(reward_components.original_reward)
                absolute_shaping_reward += abs(reward_components.shaping_reward)
                total_absolute_environment_reward += abs(
                    reward_components.original_reward
                )
                total_absolute_shaping_reward += abs(reward_components.shaping_reward)
                episode_positions.append(float(next_observation[0]))
                episode_velocities.append(float(next_observation[1]))
                episode_actions.append(action)
                reached_goal = bool(float(next_observation[0]) >= goal_position)
                if reached_goal and first_training_completion_step is None:
                    first_training_completion_step = total_interactions
                if (
                    reached_goal
                    and step_count <= 100
                    and first_training_assignment_success_step is None
                ):
                    first_training_assignment_success_step = total_interactions
                if isinstance(agent, SarsaLambdaAgent):
                    next_action = (
                        None
                        if terminated
                        else int(
                            agent.select_action(
                                np.asarray(next_observation), explore=True
                            )
                        )
                    )
                    metric = float(
                        agent.update(
                            np.asarray(observation),
                            action,
                            reward_components.training_reward,
                            np.asarray(next_observation),
                            next_action,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                        )
                    )
                elif isinstance(agent, DQNAgent):
                    returned = agent.observe(
                        np.asarray(observation),
                        action,
                        reward_components.training_reward,
                        np.asarray(next_observation),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=info,
                    )
                    metric = None if returned is None else float(returned)
                    next_action = None
                else:  # pragma: no cover - guarded by factory/type checks
                    raise TypeError(f"Unsupported learning agent: {type(agent).__name__}")

                if metric is not None:
                    if not math.isfinite(metric):
                        raise FloatingPointError(
                            f"Non-finite update metric at episode {episode_index}, "
                            f"step {step_count}: {metric}"
                        )
                    update_metrics.append(metric)
                    if isinstance(agent, DQNAgent):
                        gradient_norm = agent.last_gradient_norm
                        if gradient_norm is not None:
                            gradient_norm_metrics.append(float(gradient_norm))

                if record_trajectories:
                    transition_rows.append(
                        {
                            "condition": condition,
                            "split": "train",
                            "training_seed": training_seed,
                            "episode": episode_index,
                            "environment_seed": environment_seed,
                            "step": step_count,
                            "position": float(observation[0]),
                            "velocity": float(observation[1]),
                            "action": action,
                            "original_reward": reward_components.original_reward,
                            "shaping_reward": reward_components.shaping_reward,
                            "training_reward": reward_components.training_reward,
                            "shaped_reward": reward_components.training_reward,
                            "current_potential": reward_shaper.potential(observation),
                            "next_potential": (
                                0.0
                                if bool(terminated)
                                else reward_shaper.potential(next_observation)
                            ),
                            "next_position": float(next_observation[0]),
                            "next_velocity": float(next_observation[1]),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "epsilon_before": epsilon_before,
                            "epsilon_after": _epsilon(agent),
                            "environment_steps": total_interactions,
                            "update_metric": metric,
                            "q_value": selected_q_value,
                            "gradient_norm": (
                                agent.last_gradient_norm
                                if isinstance(agent, DQNAgent)
                                else None
                            ),
                            "device": device,
                            "config_hash": configuration_hash,
                            "git_commit": git_commit,
                        }
                    )
                observation = next_observation
                if not (terminated or truncated) and next_action is not None:
                    action = next_action
                elif not (terminated or truncated):
                    action = int(agent.select_action(np.asarray(observation), explore=True))

            agent.end_episode()
            replay_coverage = (
                agent.replay_state_coverage(coverage_bins)
                if isinstance(agent, DQNAgent) and agent.use_replay
                else None
            )
            remained_near_valley = bool(
                max(abs(value - valley_center) for value in episode_positions)
                <= valley_radius
                and max(abs(value) for value in episode_velocities)
                <= valley_velocity_threshold
            )
            episode_rows.append(
                _episode_record(
                    condition=condition,
                    training_seed=training_seed,
                    episode=episode_index,
                    environment_seed=environment_seed,
                    steps=step_count,
                    original_return=original_return,
                    shaping_return=shaping_return,
                    training_return=training_return,
                    absolute_environment_reward=absolute_environment_reward,
                    absolute_shaping_reward=absolute_shaping_reward,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    final_observation=np.asarray(observation),
                    goal_position=goal_position,
                    update_metrics=update_metrics,
                    epsilon_start=epsilon_at_episode_start,
                    positions=episode_positions,
                    velocities=episode_velocities,
                    actions=episode_actions,
                    remained_near_valley=remained_near_valley,
                    replay_coverage=replay_coverage,
                    q_values=q_value_metrics,
                    gradient_norms=gradient_norm_metrics,
                    wall_clock_seconds=time.perf_counter() - started,
                    device=device,
                    config_hash=configuration_hash,
                    git_commit=git_commit,
                    agent=agent,
                )
            )
            completed_episodes = episode_index + 1
            invocation_episodes += 1

            budget_complete = bool(
                max_environment_steps is not None
                and total_interactions >= max_environment_steps
            )
            episode_limit_complete = completed_episodes >= episode_count
            if validation_interval_steps is not None:
                should_validate = bool(
                    next_validation_step is not None
                    and total_interactions >= next_validation_step
                ) or budget_complete or episode_limit_complete
            else:
                should_validate = (
                    completed_episodes % validation_interval == 0
                    or budget_complete
                    or episode_limit_complete
                )
            if should_validate:
                evaluated = evaluate_agent(
                    agent,
                    episode_seeds=validation_seeds,
                    training_seed=training_seed,
                    condition=condition,
                    max_episode_steps=max_episode_steps,
                    capture_trajectories=True,
                )
                current_rows = [
                    _validation_record(
                        episode,
                        condition=condition,
                        training_seed=training_seed,
                        episode_index=index,
                        validation_after_episode=episode_index + 1,
                        validation_environment_steps=total_interactions,
                        checkpoint_type="periodic_validation",
                        wall_clock_seconds=time.perf_counter() - started,
                        device=device,
                        config_hash=configuration_hash,
                        git_commit=git_commit,
                    )
                    for index, episode in enumerate(evaluated)
                ]
                validation_rows.extend(current_rows)
                if record_validation_trajectories:
                    for index, episode in enumerate(evaluated):
                        validation_trajectory_rows.extend(
                            _evaluation_trajectory_records(
                                episode,
                                condition=condition,
                                training_seed=training_seed,
                                episode_index=index,
                                validation_after_episode=episode_index + 1,
                                validation_environment_steps=total_interactions,
                                config_hash=configuration_hash,
                                device=device,
                                git_commit=git_commit,
                            )
                        )
                validation_summary = summarize_episodes(current_rows)
                validation_summary["episode_length_std"] = float(
                    np.std([int(row["steps"]) for row in current_rows])
                )
                current_score = _score(validation_summary)
                history_record = {
                    "validation_after_episode": episode_index + 1,
                    "environment_steps": total_interactions,
                    "score": list(current_score),
                    **validation_summary,
                }
                validation_history.append(history_record)
                if validation_interval_steps is not None:
                    while (
                        next_validation_step is not None
                        and next_validation_step <= total_interactions
                    ):
                        next_validation_step += validation_interval_steps
                if (
                    first_validation_completion_step is None
                    and float(validation_summary["environment_completion_rate"]) > 0.0
                ):
                    first_validation_completion_step = total_interactions
                if (
                    first_validation_assignment_success_step is None
                    and float(validation_summary["assignment_success_rate"]) > 0.0
                ):
                    first_validation_assignment_success_step = total_interactions
                if best_score is None or current_score > best_score:
                    best_score = current_score
                    best_selection = dict(history_record)
                    save_agent_checkpoint(
                        best_path,
                        agent_type=agent_type,
                        agent_state=agent.state_dict(),
                        config=config,
                        training_state={
                            "completed_episodes": episode_index + 1,
                            "total_interactions": total_interactions,
                            "training_seed": training_seed,
                            "run_name": str(experiment["name"]),
                        },
                        preprocessing=preprocessing_metadata(agent),
                        selection=best_selection,
                    )

            training_state = {
                "completed_episodes": completed_episodes,
                "total_interactions": total_interactions,
                "total_absolute_environment_reward": total_absolute_environment_reward,
                "total_absolute_shaping_reward": total_absolute_shaping_reward,
                "first_training_completion_step": first_training_completion_step,
                "first_training_assignment_success_step": (
                    first_training_assignment_success_step
                ),
                "first_validation_completion_step": first_validation_completion_step,
                "first_validation_assignment_success_step": (
                    first_validation_assignment_success_step
                ),
                "best_score": None if best_score is None else list(best_score),
                "best_selection": best_selection,
                "next_validation_step": next_validation_step,
                "training_seed": training_seed,
                "run_name": str(experiment["name"]),
                "status": "in_progress",
            }
            should_snapshot = (
                completed_episodes % snapshot_interval == 0
                or should_validate
                or budget_complete
                or episode_limit_complete
                or (
                    max_episodes_this_invocation is not None
                    and invocation_episodes >= max_episodes_this_invocation
                )
            )
            if should_snapshot:
                _write_training_snapshot(
                    output_dir=output_dir,
                    checkpoint_path=latest_path,
                    agent_type=agent_type,
                    agent=agent,
                    config=config,
                    experiment=experiment,
                    episode_rows=episode_rows,
                    transition_rows=transition_rows,
                    validation_rows=validation_rows,
                    validation_trajectory_rows=validation_trajectory_rows,
                    validation_history=validation_history,
                    training_state=training_state,
                    best_selection=best_selection,
                    write_jsonl_atomic=write_jsonl_atomic,
                )
    finally:
        env.close()

    training_complete = bool(
        (
            max_environment_steps is not None
            and total_interactions >= max_environment_steps
        )
        or (max_environment_steps is None and completed_episodes >= episode_count)
    )
    final_training_state = {
        "completed_episodes": completed_episodes,
        "total_interactions": total_interactions,
        "total_absolute_environment_reward": total_absolute_environment_reward,
        "total_absolute_shaping_reward": total_absolute_shaping_reward,
        "first_training_completion_step": first_training_completion_step,
        "first_training_assignment_success_step": first_training_assignment_success_step,
        "first_validation_completion_step": first_validation_completion_step,
        "first_validation_assignment_success_step": (
            first_validation_assignment_success_step
        ),
        "best_score": None if best_score is None else list(best_score),
        "best_selection": best_selection,
        "next_validation_step": next_validation_step,
        "training_seed": training_seed,
        "run_name": str(experiment["name"]),
        "status": "complete" if training_complete else "interrupted",
    }
    _write_training_snapshot(
        output_dir=output_dir,
        checkpoint_path=latest_path,
        agent_type=agent_type,
        agent=agent,
        config=config,
        experiment=experiment,
        episode_rows=episode_rows,
        transition_rows=transition_rows,
        validation_rows=validation_rows,
        validation_trajectory_rows=validation_trajectory_rows,
        validation_history=validation_history,
        training_state=final_training_state,
        best_selection=best_selection,
        write_jsonl_atomic=write_jsonl_atomic,
    )
    if training_complete:
        for checkpoint_path in (final_path, last_path):
            save_agent_checkpoint(
                checkpoint_path,
                agent_type=agent_type,
                agent_state=agent.state_dict(),
                config=config,
                training_state=final_training_state,
                preprocessing=preprocessing_metadata(agent),
                selection=best_selection,
            )

    summary = summarize_episodes(episode_rows)
    all_metrics = [
        float(row["mean_update_metric"])
        for row in episode_rows
        if row["mean_update_metric"] is not None
    ]
    final_replay_coverage = (
        agent.replay_state_coverage(coverage_bins)
        if isinstance(agent, DQNAgent) and agent.use_replay
        else None
    )
    shaping_ratio = float(
        total_absolute_shaping_reward
        / max(total_absolute_environment_reward, 1e-12)
    )
    valley_episode_count = sum(
        bool(row["remained_near_valley"]) for row in episode_rows
    )
    diagnostics = {
        "definition": {
            "replay_coverage_bins": list(coverage_bins),
            "valley_center_position": valley_center,
            "valley_position_radius": valley_radius,
            "valley_velocity_threshold": valley_velocity_threshold,
            "valley_episode_rule": (
                "all recorded positions remain within the configured radius of "
                "the valley center and peak absolute velocity is no larger than "
                "the configured threshold"
            ),
            "shaping_dominance_ratio": shaping_dominance_ratio,
        },
        "reward_shaping": reward_shaper.metadata(),
        "first_training_environment_completion_step": first_training_completion_step,
        "first_training_assignment_success_step": (
            first_training_assignment_success_step
        ),
        "first_validation_environment_completion_step": (
            first_validation_completion_step
        ),
        "first_validation_assignment_success_step": (
            first_validation_assignment_success_step
        ),
        "episodes_remaining_near_valley": valley_episode_count,
        "episode_count": len(episode_rows),
        "final_replay_state_coverage": final_replay_coverage,
        "absolute_shaping_to_environment_ratio": shaping_ratio,
        "shaping_dominates_environment_reward": bool(
            shaping_ratio > shaping_dominance_ratio
        ),
        "evaluation_reward": "original MountainCar-v0 reward only",
        "episode_diagnostics_source": "episodes.csv",
        "phase_space_trajectory_source": "trajectories.jsonl",
    }
    summary.update(
        {
            "experiment_name": str(experiment["name"]),
            "condition": condition,
            "split": "train",
            "training_seed": training_seed,
            "agent_type": agent_type,
            "status": "complete" if training_complete else "interrupted",
            "completed_episodes": completed_episodes,
            "total_environment_interactions": total_interactions,
            "planned_environment_interactions": max_environment_steps,
            "all_update_metrics_finite": all(math.isfinite(x) for x in all_metrics),
            "best_checkpoint": str(best_path) if best_path.is_file() else None,
            "latest_checkpoint": str(latest_path),
            "final_checkpoint": str(final_path) if final_path.is_file() else None,
            "last_checkpoint": str(last_path) if last_path.is_file() else None,
            "best_validation": best_selection,
            "reward_shaping": reward_shaper.metadata(),
            "absolute_shaping_to_environment_ratio": shaping_ratio,
            "shaping_dominates_environment_reward": bool(
                shaping_ratio > shaping_dominance_ratio
            ),
            "first_training_environment_completion_step": (
                first_training_completion_step
            ),
            "first_training_assignment_success_step": (
                first_training_assignment_success_step
            ),
            "first_validation_environment_completion_step": (
                first_validation_completion_step
            ),
            "first_validation_assignment_success_step": (
                first_validation_assignment_success_step
            ),
            "episodes_remaining_near_valley": valley_episode_count,
            "final_replay_state_coverage": final_replay_coverage,
            "evaluation_reward": "original MountainCar-v0 reward only",
            "wall_clock_seconds": float(time.perf_counter() - started),
            "device": device,
            "config_hash": configuration_hash,
            "git_commit": git_commit,
        }
    )
    metadata["wall_clock_seconds"] = float(time.perf_counter() - started)
    metadata["resumed"] = resume
    metadata["status"] = "complete" if training_complete else "interrupted"
    metadata["config_hash"] = configuration_hash

    write_json(output_dir / "config.snapshot.json", dict(config))
    write_json(output_dir / "metadata.json", metadata)
    write_csv(output_dir / "episodes.csv", episode_rows, TRAIN_EPISODE_FIELDNAMES)
    write_csv(
        output_dir / "validation_episodes.csv",
        validation_rows,
        VALIDATION_EPISODE_FIELDNAMES,
    )
    write_jsonl_atomic(output_dir / "trajectories.jsonl", transition_rows)
    write_jsonl_atomic(
        output_dir / "validation_trajectories.jsonl", validation_trajectory_rows
    )
    write_json(output_dir / "validation_history.json", validation_history)
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "policy.json",
        {
            "agent_type": agent_type,
            "condition": condition,
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "final_checkpoint": str(final_path) if final_path.is_file() else None,
            "last_checkpoint": str(last_path) if last_path.is_file() else None,
            "selection_rule": [
                "assignment_success_rate",
                "environment_completion_rate",
                "lowest_median_steps_among_completions",
                "lowest_episode_length_standard_deviation",
                "earliest_checkpoint_on_exact_tie",
            ],
            "evaluation_reward": "original MountainCar-v0 reward",
        },
    )
    return summary
