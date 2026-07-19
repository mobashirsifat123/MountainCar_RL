"""Shared checkpoint-backed evaluation under the original MountainCar objective."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mountaincar_rl.analysis.statistics import summarize_episodes
from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.experiments.agent_factory import (
    build_agent,
    configured_agent_type,
    require_mapping,
)
from mountaincar_rl.experiments.evaluation import EvaluationEpisode, evaluate_agent
from mountaincar_rl.experiments.train import _prepare_output, _write_jsonl_atomic
from mountaincar_rl.utils.checkpoints import config_fingerprint, load_agent_checkpoint
from mountaincar_rl.utils.config import load_config
from mountaincar_rl.utils.logging import write_csv, write_json
from mountaincar_rl.utils.metadata import collect_run_metadata
from mountaincar_rl.utils.seeding import seed_everything


EVALUATION_FIELDNAMES = [
    "condition",
    "split",
    "training_seed",
    "episode",
    "environment_seed",
    "steps",
    "episode_return",
    "original_return",
    "shaped_return",
    "shaping_component",
    "terminated",
    "truncated",
    "environment_completion",
    "assignment_success",
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


def _episode_record(
    episode: EvaluationEpisode,
    *,
    condition: str,
    split: str,
    training_seed: int,
    episode_index: int,
    checkpoint_type: str,
    wall_clock_seconds: float,
    device: str,
    config_hash: str,
    git_commit: str | None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "split": split,
        "training_seed": training_seed,
        "episode": episode_index,
        "environment_seed": episode.environment_seed,
        "steps": episode.steps,
        "episode_return": episode.original_return,
        "original_return": episode.original_return,
        "shaped_return": episode.shaped_return,
        "shaping_component": None,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "environment_completion": episode.environment_completion,
        "assignment_success": episode.assignment_success,
        "checkpoint_type": checkpoint_type,
        # Trajectory capture is optional.  Keep scalar diagnostics available
        # when captured, while making ``--no-capture-trajectories`` a genuine
        # memory-saving mode rather than silently retaining full rollouts.
        "maximum_position": (
            max(episode.position_trajectory)
            if episode.position_trajectory
            else None
        ),
        "maximum_absolute_velocity": (
            max(abs(value) for value in episode.velocity_trajectory)
            if episode.velocity_trajectory
            else None
        ),
        "action_count_0": sum(action == 0 for action in episode.action_trajectory),
        "action_count_1": sum(action == 1 for action in episode.action_trajectory),
        "action_count_2": sum(action == 2 for action in episode.action_trajectory),
        "wall_clock_seconds": wall_clock_seconds,
        "device": device,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }


def _trajectory_records(
    episode: EvaluationEpisode,
    *,
    condition: str,
    split: str,
    training_seed: int,
    episode_index: int,
    checkpoint_type: str,
    config_hash: str,
    device: str,
    git_commit: str | None,
) -> list[dict[str, Any]]:
    rows = episode.to_transition_records(
        condition=condition,
        split=split,
        training_seed=training_seed,
        episode_index=episode_index,
    )
    for row in rows:
        row.update(
            {
                "checkpoint_type": checkpoint_type,
                "config_hash": config_hash,
                "device": device,
                "git_commit": git_commit,
            }
        )
    return rows


def _episode_seeds(
    config: Mapping[str, Any],
    *,
    split: str,
    override: Sequence[int] | None,
    episodes: int | None,
) -> list[int]:
    evaluation = require_mapping(config, "evaluation")
    seed_key = (
        "validation_episode_seeds"
        if split == "validation"
        else "final_test_episode_seeds"
    )
    if override is not None:
        seeds = [int(seed) for seed in override]
    else:
        raw = evaluation.get(seed_key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(
                f"Configuration has no {seed_key}; final-test evaluation requires "
                "the isolated --seeds-config file."
            )
        seeds = [int(seed) for seed in raw]
    if not seeds:
        raise ValueError(f"The {split} episode seed set must not be empty.")
    if episodes is not None:
        if episodes < 1 or episodes > len(seeds):
            raise ValueError(
                f"--episodes must be between 1 and the {len(seeds)} configured seeds."
            )
        seeds = seeds[:episodes]
    return seeds


def run_evaluation(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    split: str,
    episodes: int | None,
    overwrite: bool,
    checkpoint: Path | None,
    episode_seeds_override: Sequence[int] | None = None,
    capture_trajectories_override: bool | None = None,
) -> dict[str, Any]:
    """Load a policy if needed and evaluate without invoking any learning hook."""

    if split not in {"validation", "final_test"}:
        raise ValueError("split must be 'validation' or 'final_test'.")
    experiment = require_mapping(config, "experiment")
    environment = require_mapping(config, "environment")
    logging_config = require_mapping(config, "logging")
    reproducibility = require_mapping(config, "reproducibility")
    if environment.get("id") != "MountainCar-v0":
        raise ValueError("Only environment.id='MountainCar-v0' is supported.")

    agent_type = configured_agent_type(config)
    training_seed = int(experiment["training_seed"])
    condition = str(experiment["condition"])
    max_episode_steps = int(environment.get("max_episode_steps", 200))
    device = str(reproducibility.get("device", "cpu"))
    seeds = _episode_seeds(
        config,
        split=split,
        override=episode_seeds_override,
        episodes=episodes,
    )
    record_trajectories = (
        bool(logging_config.get("record_trajectories", True))
        if capture_trajectories_override is None
        else capture_trajectories_override
    )

    _prepare_output(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    configuration_hash = config_fingerprint(config)
    metadata = collect_run_metadata(dict(config), device=device)
    git_commit = metadata["git"]["commit"]
    seed_everything(
        training_seed,
        deterministic_torch=bool(reproducibility.get("deterministic_torch", True)),
    )
    construction_env = make_env(
        seed=training_seed, max_episode_steps=max_episode_steps
    )
    try:
        agent = build_agent(config, construction_env)
    finally:
        construction_env.close()

    checkpoint_payload: Mapping[str, Any] | None = None
    if agent_type != "random":
        if checkpoint is None:
            raise ValueError("Learned-agent evaluation requires a checkpoint.")
        checkpoint_payload = load_agent_checkpoint(
            checkpoint,
            map_location=device,
            expected_agent_type=agent_type,
            expected_config=config,
        )
        load_state = getattr(agent, "load_state_dict", None)
        if not callable(load_state):
            raise TypeError(f"{type(agent).__name__} cannot load checkpoint state.")
        load_state(checkpoint_payload["agent_state"])
    elif checkpoint is not None:
        raise ValueError("The random policy has no checkpoint to load.")

    evaluated = evaluate_agent(
        agent,
        episode_seeds=seeds,
        training_seed=training_seed,
        condition=condition,
        max_episode_steps=max_episode_steps,
        capture_trajectories=record_trajectories,
    )
    rows = [
        _episode_record(
            episode,
            condition=condition,
            split=split,
            training_seed=training_seed,
            episode_index=index,
            checkpoint_type=("random" if checkpoint is None else checkpoint.stem),
            wall_clock_seconds=time.perf_counter() - started,
            device=device,
            config_hash=configuration_hash,
            git_commit=git_commit,
        )
        for index, episode in enumerate(evaluated)
    ]
    trajectories: list[dict[str, Any]] = []
    if record_trajectories:
        for index, episode in enumerate(evaluated):
            trajectories.extend(
                _trajectory_records(
                    episode,
                    condition=condition,
                    split=split,
                    training_seed=training_seed,
                    episode_index=index,
                    checkpoint_type=("random" if checkpoint is None else checkpoint.stem),
                    config_hash=configuration_hash,
                    device=device,
                    git_commit=git_commit,
                )
            )

    summary = summarize_episodes(rows)
    summary.update(
        {
            "experiment_name": str(experiment["name"]),
            "condition": condition,
            "agent_type": agent_type,
            "split": split,
            "training_seed": training_seed,
            "evaluation_episode_seeds": seeds,
            "objective": "original MountainCar-v0 reward and goal",
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_type": "random" if checkpoint is None else checkpoint.stem,
            "config_hash": configuration_hash,
            "wall_clock_seconds": time.perf_counter() - started,
            "device": device,
            "git_commit": git_commit,
        }
    )
    metadata["command"] = shlex.join(sys.argv)
    metadata["evaluation_split"] = split
    metadata["evaluation_episode_seeds"] = seeds
    metadata["checkpoint"] = None if checkpoint is None else str(checkpoint)
    metadata["checkpoint_type"] = "random" if checkpoint is None else checkpoint.stem
    metadata["config_hash"] = configuration_hash
    metadata["wall_clock_seconds"] = time.perf_counter() - started

    write_json(output_dir / "config.snapshot.json", dict(config))
    write_json(output_dir / "metadata.json", metadata)
    write_csv(output_dir / "episodes.csv", rows, EVALUATION_FIELDNAMES)
    _write_jsonl_atomic(output_dir / "trajectories.jsonl", trajectories)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "policy.json",
        {
            "agent_type": agent_type,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_selection": (
                None if checkpoint_payload is None else checkpoint_payload.get("selection")
            ),
            "evaluation_mode": "greedy; no learning/replay/optimizer mutation",
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "final_test"), default="validation"
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--seeds-config",
        type=Path,
        help="Required isolated seed file for --split final_test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--capture-trajectories",
        dest="capture_trajectories",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-capture-trajectories",
        dest="capture_trajectories",
        action="store_false",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed_override: Sequence[int] | None = None
    if args.split == "final_test":
        if args.seeds_config is None:
            raise ValueError("--split final_test requires --seeds-config.")
        seeds_config = load_config(args.seeds_config)
        if seeds_config.get("split") != "final_test":
            raise ValueError("--seeds-config must declare split: final_test.")
        raw = seeds_config.get("episode_seeds")
        if not isinstance(raw, list):
            raise ValueError("--seeds-config episode_seeds must be a list.")
        seed_override = [int(seed) for seed in raw]
    elif args.seeds_config is not None:
        raise ValueError("--seeds-config is only valid with --split final_test.")

    logging_config = require_mapping(config, "logging")
    base_output = Path(str(logging_config["output_dir"]))
    default_output = base_output.parent / f"{base_output.name}_{args.split}"
    checkpoint = args.checkpoint
    if checkpoint is None and configured_agent_type(config) != "random":
        checkpoint_dir = logging_config.get("checkpoint_dir")
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
            raise ValueError("Learning configuration has no logging.checkpoint_dir.")
        checkpoint = Path(checkpoint_dir) / "best.pt"

    summary = run_evaluation(
        config,
        output_dir=args.output_dir or default_output,
        split=args.split,
        episodes=args.episodes,
        overwrite=bool(args.overwrite),
        checkpoint=checkpoint,
        episode_seeds_override=seed_override,
        capture_trajectories_override=args.capture_trajectories,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
