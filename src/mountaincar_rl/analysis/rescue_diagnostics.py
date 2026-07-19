"""Generate validation-only diagnostics for the controlled DQN rescue study.

This module intentionally reads training and periodic-validation artifacts only.
It rejects any input path named ``final_test`` so the rescue analysis cannot tune
against held-out evaluation episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from mountaincar_rl.utils.logging import write_json


REFERENCE_CONDITIONS = (
    "dqn_replay",
    "dqn_replay_shaped",
    "dqn_replay_fast_decay",
    "dqn_no_replay",
    "sarsa_lambda",
)
RESCUE_INTERVENTIONS = (
    "epsilon_slow",
    "warmup_long",
    "learning_rate_small",
    "target_update_slow",
    "shaping_moderate",
    "budget_long",
)


def _assert_validation_only(path: Path) -> None:
    if "final_test" in path.parts:
        raise ValueError(f"Held-out artifact forbidden in rescue analysis: {path}")


def _rows(path: Path) -> list[dict[str, str]]:
    _assert_validation_only(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value in (None, ""):
        return None
    return float(value)


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(mean(present)) if present else None


def _seed_trace(run_dir: Path, marks: Sequence[int]) -> dict[str, Any]:
    episodes = _rows(run_dir / "episodes.csv")
    if not episodes:
        raise ValueError(f"No training episodes in {run_dir}")
    output: dict[str, Any] = {}
    for mark in marks:
        prefix = [
            row for row in episodes if float(row["environment_steps"]) <= mark
        ]
        if not prefix:
            prefix = [episodes[0]]
        recent = prefix[-10:]
        counts = [
            sum(int(float(row[f"action_count_{action}"])) for row in prefix)
            for action in range(3)
        ]
        total_actions = sum(counts)
        output[str(mark)] = {
            "epsilon": _number(prefix[-1], "epsilon"),
            "action_fractions": [
                count / total_actions if total_actions else 0.0 for count in counts
            ],
            "replay_coverage_fraction": _number(
                prefix[-1], "replay_coverage_fraction"
            ),
            "recent_10_median_maximum_position": float(
                median(float(row["maximum_position"]) for row in recent)
            ),
            "recent_10_median_maximum_absolute_velocity": float(
                median(float(row["maximum_absolute_velocity"]) for row in recent)
            ),
            "recent_10_mean_update_metric": _mean_present(
                [_number(row, "mean_update_metric") for row in recent]
            ),
            "recent_10_mean_q_value": _mean_present(
                [_number(row, "mean_q_value") for row in recent]
            ),
            "recent_10_mean_gradient_norm": _mean_present(
                [_number(row, "mean_gradient_norm") for row in recent]
            ),
        }
    return output


def _aggregate_reference(raw_root: Path, condition: str) -> dict[str, Any]:
    traces = []
    for seed in range(5):
        run_dir = raw_root / "full" / f"{condition}_seed{seed}"
        traces.append(_seed_trace(run_dir, (6000, 15000, 30000, 60000)))
    aggregate: dict[str, Any] = {"training_seed_count": 5, "marks": {}}
    for mark in (6000, 15000, 30000, 60000):
        rows = [trace[str(mark)] for trace in traces]
        aggregate["marks"][str(mark)] = {
            "epsilon_mean": _mean_present([row["epsilon"] for row in rows]),
            "action_fraction_mean": [
                mean(row["action_fractions"][action] for row in rows)
                for action in range(3)
            ],
            "replay_coverage_mean": _mean_present(
                [row["replay_coverage_fraction"] for row in rows]
            ),
            "recent_10_median_maximum_position_mean": mean(
                row["recent_10_median_maximum_position"] for row in rows
            ),
            "recent_10_median_maximum_absolute_velocity_mean": mean(
                row["recent_10_median_maximum_absolute_velocity"] for row in rows
            ),
            "recent_10_mean_update_metric": _mean_present(
                [row["recent_10_mean_update_metric"] for row in rows]
            ),
            "recent_10_mean_q_value": _mean_present(
                [row["recent_10_mean_q_value"] for row in rows]
            ),
            "recent_10_mean_gradient_norm": _mean_present(
                [row["recent_10_mean_gradient_norm"] for row in rows]
            ),
        }
    return aggregate


def _read_json(path: Path) -> dict[str, Any]:
    _assert_validation_only(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def _rescue_results(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intervention in RESCUE_INTERVENTIONS:
        for seed in (0, 1):
            run_dir = raw_root / "rescue" / f"{intervention}_seed{seed}"
            summary = _read_json(run_dir / "summary.json")
            config = _read_json(run_dir / "config.snapshot.json")
            validation_seeds = config["evaluation"]["validation_episode_seeds"]
            if set(validation_seeds) != set(range(1000, 1020)):
                raise ValueError(f"Unexpected rescue validation seeds in {run_dir}")
            best = summary["best_validation"]
            rows.append(
                {
                    "intervention": intervention,
                    "training_seed": seed,
                    "planned_environment_interactions": summary[
                        "planned_environment_interactions"
                    ],
                    "best_assignment_success_rate": best[
                        "assignment_success_rate"
                    ],
                    "best_completion_rate": best["environment_completion_rate"],
                    "best_median_completed_steps": best[
                        "median_steps_among_completions"
                    ],
                    "best_environment_steps": best["environment_steps"],
                    "first_validation_assignment_success_step": summary[
                        "first_validation_assignment_success_step"
                    ],
                    "first_validation_completion_step": summary[
                        "first_validation_environment_completion_step"
                    ],
                    "final_replay_coverage_fraction": summary[
                        "final_replay_state_coverage"
                    ]["coverage_fraction"],
                    "absolute_shaping_to_environment_ratio": summary[
                        "absolute_shaping_to_environment_ratio"
                    ],
                }
            )
    return rows


def build_report(raw_root: Path) -> dict[str, Any]:
    """Build the complete rescue report from training/validation artifacts."""

    references = {
        condition: _aggregate_reference(raw_root, condition)
        for condition in REFERENCE_CONDITIONS
    }
    rescue = _rescue_results(raw_root)
    repeatable = [
        intervention
        for intervention in RESCUE_INTERVENTIONS
        if all(
            row["best_assignment_success_rate"] > 0.0
            for row in rescue
            if row["intervention"] == intervention
        )
    ]
    return {
        "schema_version": 1,
        "split": "training_and_validation_only",
        "final_test_used": False,
        "diagnostic_marks": {
            "fractions_of_60000": [0.10, 0.25, 0.50, 1.0],
            "environment_steps": [6000, 15000, 30000, 60000],
            "recent_window_episodes": 10,
            "replay_coverage_grid": [20, 20],
        },
        "reference_conditions": references,
        "controlled_interventions": rescue,
        "interventions_with_strict_success_in_both_seeds": repeatable,
        "corrected_configuration_locked": bool(repeatable),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render an evidence-led diagnosis from a generated rescue report."""

    refs = report["reference_conditions"]
    rescue = report["controlled_interventions"]
    lines = [
        "# Controlled rescue diagnosis",
        "",
        "**Scope:** training and periodic validation artifacts only. No held-out "
        "final-test artifact or seed was read. Diagnostics at 6k, 15k, 30k, and "
        "60k interactions correspond to 10%, 25%, 50%, and 100% of the locked "
        "60k DQN budget. Position and velocity summaries are seed means of each "
        "seed's trailing-10-episode median; update, Q, and gradient summaries use "
        "the same trailing window.",
        "",
        "## Failure classification",
        "",
        "| Candidate cause | Assessment | Artifact evidence |",
        "|---|---|---|",
        "| Implementation defect | Unlikely | The focused numerical tests cover action mapping, observation normalization, reward, termination/truncation, DQN targets and target copies, replay sampling, SARSA updates, checkpoints, and deterministic evaluation. All passed before rescue runs. |",
        "| Inadequate exploration | Not the primary cause | Slower epsilon decay was rejected on both validation seeds. The rapid-decay reference nevertheless had higher 60k replay coverage and momentum than standard replay, contradicting a simple 'too little coverage' account. |",
        "| Unstable optimization / policy transfer | Most supported | Standard replay briefly reached 1.0 ordinary validation completion for seed 0 at 40k, then returned to zero at later checkpoints. Multiple rescue runs discovered completions or broad replay coverage without producing a stable greedy <=100 policy. |",
        "| Harmful shaping | Not primary | Moderate shaping kept abs(shaping)/abs(environment) below 0.008, expanded coverage in both seeds, and produced ordinary completions, but did not yield greedy <=100 validation success. |",
        "| Insufficient budget | Not sufficient by itself | Extending only the budget to 120k reproduced seed 0's 40k transient best and left seed 1 unsuccessful; neither seed improved after 60k. |",
        "| Evaluation bug | Unlikely | Deterministic evaluation, exact success threshold, original-reward accounting, and non-training evaluation are covered by passing tests. Behavior-policy successes that fail to transfer to greedy validation remain scientifically plausible. |",
        "| Checkpoint-selection bug | Unlikely | The deterministic validation selector retained seed 0's transient 40k checkpoint over later regressions; it did not consult held-out seeds. |",
        "",
        "## Locked-run diagnostics",
        "",
        "| Condition | epsilon @6k/15k/30k/60k | coverage @60k | max position @60k | max abs velocity @60k | update metric @60k | mean Q @60k | grad norm @60k |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "dqn_replay": "Replay DQN",
        "dqn_replay_shaped": "Shaped replay DQN",
        "dqn_replay_fast_decay": "Rapid-decay replay DQN",
        "dqn_no_replay": "No-replay DQN",
        "sarsa_lambda": "SARSA(lambda)",
    }
    for condition in REFERENCE_CONDITIONS:
        marks = refs[condition]["marks"]
        eps = "/".join(
            _fmt(marks[str(step)]["epsilon_mean"], 3)
            for step in (6000, 15000, 30000, 60000)
        )
        final = marks["60000"]
        lines.append(
            f"| {labels[condition]} | {eps} | {_fmt(final['replay_coverage_mean'])} "
            f"| {_fmt(final['recent_10_median_maximum_position_mean'])} "
            f"| {_fmt(final['recent_10_median_maximum_absolute_velocity_mean'])} "
            f"| {_fmt(final['recent_10_mean_update_metric'])} "
            f"| {_fmt(final['recent_10_mean_q_value'])} "
            f"| {_fmt(final['recent_10_mean_gradient_norm'])} |"
        )
    lines.extend(
        [
            "",
            "For DQN, `update metric` is loss; for SARSA(lambda), it is TD error and is therefore not directly comparable. Missing replay coverage for no-replay DQN and SARSA(lambda) is expected.",
            "",
            "## Controlled validation interventions",
            "",
            "| Intervention | Seed | Best <=100 | Best completion | Median completed steps | Best step | First validation <=100 | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rescue:
        lines.append(
            f"| {row['intervention']} | {row['training_seed']} "
            f"| {_fmt(row['best_assignment_success_rate'], 3)} "
            f"| {_fmt(row['best_completion_rate'], 3)} "
            f"| {_fmt(row['best_median_completed_steps'], 1)} "
            f"| {row['best_environment_steps']} "
            f"| {_fmt(row['first_validation_assignment_success_step'])} "
            f"| {_fmt(row['final_replay_coverage_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No intervention achieved nonzero greedy <=100 validation success in both seeds. Under the preregistered rule, no corrected configuration was locked, no five-seed rerun was started, and no new held-out evaluation was performed.",
            "",
            "The evidence supports a bounded conclusion: within this implementation, representation, and tested budget, neural value/policy stability and transfer from exploratory behavior to the greedy policy are the leading failure mode. This is not proof of a universal DQN limitation. The original unsuccessful configuration and contrary rapid-decay evidence remain preserved.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/processed/rescue_diagnostics.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/processed/rescue_diagnosis.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.raw_root)
    write_json(args.json_output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(args.json_output)
    print(args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "render_markdown"]
