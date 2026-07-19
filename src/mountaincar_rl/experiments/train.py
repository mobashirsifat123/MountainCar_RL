"""Configuration-driven training CLI for random, SARSA(lambda), and DQN."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mountaincar_rl.agents.random_agent import RandomAgent
from mountaincar_rl.analysis.statistics import summarize_episodes
from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.experiments.rollout import (
    EPISODE_FIELDNAMES,
    rollout_episodes,
    training_episode_seeds,
)
from mountaincar_rl.experiments.training import run_learning_training
from mountaincar_rl.utils.config import load_config
from mountaincar_rl.utils.logging import write_csv, write_json
from mountaincar_rl.utils.metadata import collect_run_metadata
from mountaincar_rl.utils.seeding import seed_everything


ARTIFACT_NAMES = (
    "config.snapshot.json",
    "metadata.json",
    "episodes.csv",
    "trajectories.jsonl",
    "summary.json",
    "policy.json",
    "validation_episodes.csv",
    "validation_trajectories.jsonl",
    "validation_history.json",
    "diagnostics.json",
    "run_state.json",
)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")
    return value


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _prepare_output(directory: Path, *, overwrite: bool) -> None:
    conflicts = [name for name in ARTIFACT_NAMES if (directory / name).exists()]
    if conflicts and not overwrite:
        joined = ", ".join(conflicts)
        raise FileExistsError(
            f"Output directory {directory} already contains run artifacts: {joined}. "
            "Choose another --output-dir or pass --overwrite."
        )
    directory.mkdir(parents=True, exist_ok=True)


def run_random_training(
    config: Mapping[str, Any], *, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    """Execute and persist a deterministic random-policy training baseline."""

    experiment = _mapping(config, "experiment")
    environment = _mapping(config, "environment")
    agent_config = _mapping(config, "agent")
    logging_config = _mapping(config, "logging")
    reproducibility = _mapping(config, "reproducibility")

    if agent_config.get("type") != "random":
        raise ValueError("run_random_training requires agent.type='random'.")
    if environment.get("id") != "MountainCar-v0":
        raise ValueError("Only environment.id='MountainCar-v0' is supported.")

    training_seed = int(experiment["training_seed"])
    episode_count = int(experiment["episodes"])
    max_episode_steps = int(environment.get("max_episode_steps", 200))
    deterministic_torch = bool(reproducibility.get("deterministic_torch", True))
    record_trajectories = bool(logging_config.get("record_trajectories", True))
    condition = str(experiment.get("condition", "random"))

    _prepare_output(output_dir, overwrite=overwrite)
    seed_everything(training_seed, deterministic_torch=deterministic_torch)
    env = make_env(seed=training_seed, max_episode_steps=max_episode_steps)
    try:
        agent = RandomAgent(env.action_space, seed=training_seed)
        episode_rows, transition_rows = rollout_episodes(
            env=env,
            agent=agent,
            condition=condition,
            split="train",
            training_seed=training_seed,
            episode_seeds=training_episode_seeds(training_seed, episode_count),
            record_trajectories=record_trajectories,
            explore=True,
        )
    finally:
        env.close()

    summary = summarize_episodes(episode_rows)
    summary.update(
        {
            "experiment_name": str(experiment["name"]),
            "condition": condition,
            "split": "train",
            "training_seed": training_seed,
            "metric_note": "Rates use original MountainCar termination and return.",
        }
    )
    metadata = collect_run_metadata(dict(config), device="cpu")
    metadata["command"] = shlex.join(sys.argv)

    write_json(output_dir / "config.snapshot.json", dict(config))
    write_json(output_dir / "metadata.json", metadata)
    write_csv(output_dir / "episodes.csv", episode_rows, EPISODE_FIELDNAMES)
    _write_jsonl_atomic(output_dir / "trajectories.jsonl", transition_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "policy.json",
        {
            "agent_type": "random",
            "training_seed": training_seed,
            "trainable_parameters": 0,
            "checkpoint": None,
            "note": "Sanity-check policy; no learned weights exist.",
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override logging.output_dir with an explicit run directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace known artifacts in an existing run directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from logging.checkpoint_dir/latest.pt and durable metrics.",
    )
    parser.add_argument(
        "--max-episodes-this-invocation",
        type=int,
        help="Controlled interruption hook used to verify durable resumption.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    logging_config = _mapping(config, "logging")
    output_dir = args.output_dir or Path(str(logging_config["output_dir"]))
    agent_config = _mapping(config, "agent")
    if agent_config.get("type") == "random":
        if args.resume or args.max_episodes_this_invocation is not None:
            raise ValueError("Random-policy runs do not support resume controls.")
        summary = run_random_training(
            config, output_dir=output_dir, overwrite=bool(args.overwrite)
        )
    else:
        if not args.resume:
            _prepare_output(output_dir, overwrite=bool(args.overwrite))
        checkpoint_dir_value = logging_config.get("checkpoint_dir")
        if not isinstance(checkpoint_dir_value, str) or not checkpoint_dir_value:
            raise ValueError(
                "Learning configurations require logging.checkpoint_dir."
            )
        summary = run_learning_training(
            config,
            output_dir=output_dir,
            checkpoint_dir=Path(checkpoint_dir_value),
            overwrite=bool(args.overwrite),
            write_jsonl_atomic=_write_jsonl_atomic,
            resume=bool(args.resume),
            max_episodes_this_invocation=args.max_episodes_this_invocation,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
