"""Run a bounded, validation-only sweep and preserve every attempted trial."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from mountaincar_rl.experiments.training import run_learning_training
from mountaincar_rl.experiments.train import _write_jsonl_atomic
from mountaincar_rl.utils.config import load_config
from mountaincar_rl.utils.logging import write_json


HELD_OUT_SEEDS = frozenset(range(2000, 2020))


def validation_score(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return the preregistered lexicographic score for a trial's best checkpoint."""

    best = summary.get("best_validation")
    if not isinstance(best, Mapping):
        raise ValueError("Completed tuning run has no best_validation mapping")
    median = best.get("median_steps_among_completions")
    stability = float(best.get("episode_length_std", float("inf")))
    return (
        float(best["assignment_success_rate"]),
        float(best["environment_completion_rate"]),
        -float(median) if median is not None else -1.0e12,
        -stability,
    )


def _validate_tuning_config(config: Mapping[str, Any], path: Path) -> None:
    evaluation = config.get("evaluation")
    experiment = config.get("experiment")
    if not isinstance(evaluation, Mapping) or not isinstance(experiment, Mapping):
        raise ValueError(f"Tuning config lacks evaluation/experiment mapping: {path}")
    seeds = evaluation.get("validation_episode_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Tuning config lacks validation seeds: {path}")
    overlap = sorted({int(value) for value in seeds} & HELD_OUT_SEEDS)
    if overlap:
        raise ValueError(f"Held-out seeds appear in tuning config {path}: {overlap}")
    if int(experiment.get("training_seed", -1)) != 0:
        raise ValueError(f"Bounded tuning is fixed to training seed 0: {path}")


def run_sweep(config_paths: Sequence[Path], *, overwrite: bool) -> dict[str, Any]:
    """Execute all declared trials and rank generated validation summaries."""

    if not config_paths:
        raise ValueError("At least one --config is required")
    rows: list[dict[str, Any]] = []
    for path in config_paths:
        config = load_config(path)
        _validate_tuning_config(config, path)
        logging = config.get("logging")
        if not isinstance(logging, Mapping):
            raise ValueError(f"Tuning config lacks logging mapping: {path}")
        summary = run_learning_training(
            config,
            output_dir=Path(str(logging["output_dir"])),
            checkpoint_dir=Path(str(logging["checkpoint_dir"])),
            overwrite=overwrite,
            write_jsonl_atomic=_write_jsonl_atomic,
        )
        if summary.get("status") != "complete":
            raise RuntimeError(f"Tuning run did not complete: {path}")
        diagnostics_path = Path(str(config["logging"]["output_dir"])) / "diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        score = validation_score(summary)
        rows.append(
            {
                "config": path.as_posix(),
                "condition": summary["condition"],
                "config_hash": summary["config_hash"],
                "score": list(score),
                "best_validation": summary["best_validation"],
                "first_validation_environment_completion_step": summary[
                    "first_validation_environment_completion_step"
                ],
                "first_validation_assignment_success_step": summary[
                    "first_validation_assignment_success_step"
                ],
                "absolute_shaping_to_environment_ratio": summary[
                    "absolute_shaping_to_environment_ratio"
                ],
                "shaping_dominates_environment_reward": summary[
                    "shaping_dominates_environment_reward"
                ],
                "final_replay_state_coverage": diagnostics.get(
                    "final_replay_state_coverage"
                ),
                "total_environment_interactions": summary[
                    "total_environment_interactions"
                ],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "output_dir": str(config["logging"]["output_dir"]),
            }
        )
    ranked = sorted(rows, key=lambda row: tuple(row["score"]), reverse=True)
    return {
        "schema_version": 1,
        "split": "validation",
        "final_test_used": False,
        "selection_rule": [
            "assignment_success_rate_desc",
            "environment_completion_rate_desc",
            "median_completed_steps_asc",
            "episode_length_std_asc",
        ],
        "generated_at": datetime.now().astimezone().isoformat(),
        "trials": rows,
        "ranking": [row["config"] for row in ranked],
        "selected": ranked[0],
    }


def _append_experiment_log(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "",
        f"## Bounded validation sweep — {report['generated_at']}",
        "",
        "Generated from actual run artifacts; final-test seeds were rejected by code.",
        "",
        "| Rank | Config | <=100 rate | Completion | Median completed steps | Stability SD | Interactions | Seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    trials = {row["config"]: row for row in report["trials"]}
    for rank, name in enumerate(report["ranking"], start=1):
        row = trials[name]
        best = row["best_validation"]
        lines.append(
            f"| {rank} | `{name}` | {best['assignment_success_rate']:.3f} | "
            f"{best['environment_completion_rate']:.3f} | "
            f"{best['median_steps_among_completions']} | "
            f"{best['episode_length_std']:.3f} | "
            f"{row['total_environment_interactions']} | {row['wall_clock_seconds']:.2f} |"
        )
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment-log", type=Path, default=Path("experiments/experiment_log.md")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_sweep(args.config, overwrite=args.overwrite)
    write_json(args.output, report)
    _append_experiment_log(args.experiment_log, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_sweep", "validation_score"]
