"""Artifact-derived aggregation for validation-only intervention runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mountaincar_rl.utils.logging import write_json


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required run artifact not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"Run artifact must contain a mapping: {path}")
    return dict(value)


def aggregate_intervention_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Extract comparable shaping/failure metrics without manual transcription.

    The selected-validation row is the training runner's preregistered best
    checkpoint selection, based only on original-objective validation metrics.
    This helper rejects any run snapshot that contains a held-out seed in the
    validation list.
    """

    if not run_dirs:
        raise ValueError("At least one run directory is required")
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _read_mapping(run_dir / "summary.json")
        diagnostics = _read_mapping(run_dir / "diagnostics.json")
        config = _read_mapping(run_dir / "config.snapshot.json")
        evaluation = config.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"Run {run_dir} lacks evaluation configuration")
        seeds = evaluation.get("validation_episode_seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Run {run_dir} lacks validation episode seeds")
        held_out_overlap = sorted(set(int(seed) for seed in seeds) & set(range(2000, 2020)))
        if held_out_overlap:
            raise ValueError(
                f"Run {run_dir} used held-out seeds during validation: {held_out_overlap}"
            )
        best = summary.get("best_validation")
        if not isinstance(best, Mapping):
            raise ValueError(f"Run {run_dir} lacks best_validation metrics")
        agent = config.get("agent")
        reward = config.get("reward_shaping")
        diagnostic_config = config.get("diagnostics")
        if not isinstance(agent, Mapping) or not isinstance(reward, Mapping):
            raise ValueError(f"Run {run_dir} lacks intervention configuration")
        if not isinstance(diagnostic_config, Mapping):
            raise ValueError(f"Run {run_dir} lacks diagnostic configuration")
        epsilon = agent.get("epsilon")
        if not isinstance(epsilon, Mapping):
            raise ValueError(f"Run {run_dir} lacks epsilon configuration")
        dominance_threshold = float(diagnostic_config["shaping_dominance_ratio"])
        shaping_ratio = float(
            diagnostics["absolute_shaping_to_environment_ratio"]
        )
        rows.append(
            {
                "run_dir": run_dir.as_posix(),
                "condition": summary["condition"],
                "training_seed": int(summary["training_seed"]),
                "beta": float(reward["beta"]) if bool(reward["enabled"]) else 0.0,
                "eta": float(reward["eta"]),
                "energy_scale": float(reward["energy_scale"]),
                "epsilon_decay_steps": int(epsilon["decay_steps"]),
                "replay_warmup_steps": int(agent["warmup_steps"]),
                "first_training_environment_completion_step": summary[
                    "first_training_environment_completion_step"
                ],
                "first_training_assignment_success_step": summary[
                    "first_training_assignment_success_step"
                ],
                "first_validation_environment_completion_step": summary[
                    "first_validation_environment_completion_step"
                ],
                "first_validation_assignment_success_step": summary[
                    "first_validation_assignment_success_step"
                ],
                "selected_validation_after_episode": best[
                    "validation_after_episode"
                ],
                "selected_validation_environment_steps": best["environment_steps"],
                "selected_validation_completion_rate": best[
                    "environment_completion_rate"
                ],
                "selected_validation_assignment_success_rate": best[
                    "assignment_success_rate"
                ],
                "selected_validation_mean_steps_among_completions": best[
                    "mean_steps_among_completions"
                ],
                "selected_validation_median_steps_among_completions": best[
                    "median_steps_among_completions"
                ],
                "selected_validation_mean_original_return": best[
                    "mean_episode_return"
                ],
                "absolute_shaping_to_environment_ratio": shaping_ratio,
                "shaping_dominance_threshold": dominance_threshold,
                "rejected_for_numeric_reward_dominance": bool(
                    shaping_ratio > dominance_threshold
                ),
                "final_replay_coverage_fraction": diagnostics[
                    "final_replay_state_coverage"
                ]["coverage_fraction"],
                "episodes_remaining_near_valley": diagnostics[
                    "episodes_remaining_near_valley"
                ],
                "training_episodes": diagnostics["episode_count"],
                "evaluation_reward": diagnostics["evaluation_reward"],
            }
        )
    return {
        "schema_version": 1,
        "split": "validation",
        "training_seed_count": len({row["training_seed"] for row in rows}),
        "single_seed_warning": (
            "These validation-only observations cannot establish reliability or "
            "support final hypothesis claims."
        ),
        "selection_basis": "periodic validation under original MountainCar-v0 objective",
        "final_test_used": False,
        "conditions": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate_intervention_runs(args.run_dir)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_intervention_runs"]
