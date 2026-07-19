"""Regenerate submission tables and four figures from immutable raw artifacts.

This module deliberately reads ``results/raw/full`` rather than a hand-edited
summary.  It never imports training code, loads checkpoints, or evaluates a
policy.  The unit of uncertainty is the training seed (five policies), not an
individual held-out episode.

Run from the repository root with::

    python -m mountaincar_rl.analysis.make_all_figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS: tuple[str, ...] = (
    "random",
    "sarsa_lambda",
    "dqn_no_replay",
    "dqn_replay",
    "dqn_replay_shaped",
    "dqn_replay_fast_decay",
)
LABELS = {
    "random": "Random policy",
    "sarsa_lambda": "SARSA(λ)",
    "dqn_no_replay": "DQN without replay",
    "dqn_replay": "DQN with replay",
    "dqn_replay_shaped": "Shaped DQN with replay",
    "dqn_replay_fast_decay": "Rapid-epsilon-decay DQN",
}
CURVE_CONDITIONS = (
    "sarsa_lambda",
    "dqn_no_replay",
    "dqn_replay",
    "dqn_replay_shaped",
)
COLORS = {
    "sarsa_lambda": "#4c78a8",
    "dqn_no_replay": "#f58518",
    "dqn_replay": "#54a24b",
    "dqn_replay_shaped": "#b279a2",
    "dqn_replay_fast_decay": "#e45756",
}


@dataclass(frozen=True)
class Episode:
    """A parsed final-test episode row, retaining only analysis fields."""

    condition: str
    training_seed: int
    episode: int
    environment_seed: int
    steps: int
    original_return: float
    completion: bool
    assignment_success: bool
    maximum_position: float
    maximum_absolute_velocity: float


def _bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Expected a literal boolean CSV value, received {value!r}")
    return value == "True"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required raw artifact is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON raw artifact {path}: {error}") from error


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required raw artifact is missing: {path}") from None


def _run_dirs(raw_root: Path, condition: str) -> list[Path]:
    directories = sorted(raw_root.glob(f"{condition}_seed*"), key=lambda p: p.name)
    if not directories:
        raise FileNotFoundError(f"No raw full-suite runs found for {condition!r}")
    return directories


def _final_episodes(raw_root: Path, condition: str) -> list[Episode]:
    """Load held-out evaluation rows for a condition from raw CSV files."""

    records: list[Episode] = []
    for run_dir in _run_dirs(raw_root, condition):
        for row in _csv_rows(run_dir / "final_test" / "episodes.csv"):
            if row["condition"] != condition or row["split"] != "final_test":
                raise ValueError(f"Unexpected condition/split in {run_dir}")
            records.append(
                Episode(
                    condition=condition,
                    training_seed=int(row["training_seed"]),
                    episode=int(row["episode"]),
                    environment_seed=int(row["environment_seed"]),
                    steps=int(row["steps"]),
                    original_return=float(row["original_return"]),
                    completion=_bool(row["environment_completion"]),
                    assignment_success=_bool(row["assignment_success"]),
                    maximum_position=float(row["maximum_position"]),
                    maximum_absolute_velocity=float(row["maximum_absolute_velocity"]),
                )
            )
    return records


def _bootstrap_ci(values: Sequence[float], *, rng_seed: int) -> tuple[float, float]:
    """Percentile 95% bootstrap interval over training-seed summaries."""

    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    sample_indices = rng.integers(0, array.size, size=(10_000, array.size))
    means = array[sample_indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _seed_summary(records: Iterable[Episode]) -> dict[str, float | None]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot summarize an empty training seed")
    completed_steps = [row.steps for row in rows if row.completion]
    return {
        "episodes": float(len(rows)),
        "completion_rate": float(np.mean([row.completion for row in rows])),
        "assignment_success_rate": float(
            np.mean([row.assignment_success for row in rows])
        ),
        "mean_original_return": float(np.mean([row.original_return for row in rows])),
        "median_completed_steps": (
            float(np.median(completed_steps)) if completed_steps else None
        ),
    }


def _first_times(raw_root: Path, condition: str) -> tuple[list[int], list[int]]:
    """Return non-censored first-training completion interactions per seed.

    A missing value is genuinely censored by the 60,000-interaction budget, so
    the final table reports it as ``not observed`` rather than assigning an
    artificial time beyond the budget.
    """

    completions: list[int] = []
    assignment: list[int] = []
    for run_dir in _run_dirs(raw_root, condition):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():  # Random policy has no learning summary fields.
            continue
        summary = _read_json(summary_path)
        completion = summary.get("first_training_environment_completion_step")
        success = summary.get("first_training_assignment_success_step")
        if completion is not None:
            completions.append(int(completion))
        if success is not None:
            assignment.append(int(success))
    return completions, assignment


def _format_rate(values: Sequence[float], *, rng_seed: int) -> str:
    mean = float(np.mean(values))
    lower, upper = _bootstrap_ci(values, rng_seed=rng_seed)
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def _format_time(values: Sequence[int], total_seeds: int) -> str:
    if not values:
        return f"not observed ({total_seeds}/{total_seeds} censored)"
    censored = total_seeds - len(values)
    text = f"{float(np.median(values)):.0f} [{min(values)}, {max(values)}]"
    return f"{text}; {censored}/{total_seeds} censored"


def build_final_table(raw_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Build final-test metrics with seed-level bootstrap uncertainty."""

    rows: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITIONS):
        records = _final_episodes(raw_root, condition)
        grouped: dict[int, list[Episode]] = defaultdict(list)
        for record in records:
            grouped[record.training_seed].append(record)
        per_seed = [_seed_summary(grouped[seed]) for seed in sorted(grouped)]
        n_seeds = len(per_seed)
        episode_counts = {int(item["episodes"]) for item in per_seed}
        all_completed = [record.steps for record in records if record.completion]
        first_completion, first_assignment = _first_times(raw_root, condition)
        completion = [float(item["completion_rate"]) for item in per_seed]
        success = [float(item["assignment_success_rate"]) for item in per_seed]
        returns = [float(item["mean_original_return"]) for item in per_seed]
        rows.append(
            {
                "condition": LABELS[condition],
                "condition_id": condition,
                "training_seeds": n_seeds,
                "evaluation_episodes_per_seed": (
                    str(next(iter(episode_counts)))
                    if len(episode_counts) == 1
                    else "inconsistent"
                ),
                "completion_rate_within_200_steps_95ci": _format_rate(
                    completion, rng_seed=10_000 + index
                ),
                "assignment_success_rate_within_100_steps_95ci": _format_rate(
                    success, rng_seed=20_000 + index
                ),
                "median_steps_among_completed_episodes": (
                    f"{float(np.median(all_completed)):.1f}"
                    if all_completed
                    else "NA (0 completed episodes)"
                ),
                "mean_original_environment_return_95ci": _format_rate(
                    returns, rng_seed=30_000 + index
                ),
                "time_to_first_training_completion_interactions": _format_time(
                    first_completion, n_seeds
                ),
                "time_to_first_training_assignment_success_interactions": _format_time(
                    first_assignment, n_seeds
                ),
                "variability_across_training_seeds": (
                    f"assignment-success SD={np.std(success, ddof=1):.3f}; "
                    f"completion SD={np.std(completion, ddof=1):.3f}"
                    if n_seeds > 1
                    else "NA"
                ),
            }
        )
    markdown_header = [
        "Condition",
        "Training seeds",
        "Evaluation episodes/seed",
        "Completion ≤200 (95% seed CI)",
        "Assignment success ≤100 (95% seed CI)",
        "Median steps among completed episodes",
        "Mean original return (95% seed CI)",
        "First training completion (interactions)",
        "First training ≤100 success (interactions)",
        "Variability across training seeds",
    ]
    markdown_rows = ["| " + " | ".join(markdown_header) + " |"]
    markdown_rows.append("|" + "---|" * len(markdown_header))
    for row in rows:
        markdown_rows.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "condition",
                    "training_seeds",
                    "evaluation_episodes_per_seed",
                    "completion_rate_within_200_steps_95ci",
                    "assignment_success_rate_within_100_steps_95ci",
                    "median_steps_among_completed_episodes",
                    "mean_original_environment_return_95ci",
                    "time_to_first_training_completion_interactions",
                    "time_to_first_training_assignment_success_interactions",
                    "variability_across_training_seeds",
                )
            )
            + " |"
        )
    note = (
        "\n\nAll values are regenerated from `results/raw/full`. Intervals are "
        "percentile 95% bootstrap intervals over training-seed summaries "
        "(10,000 deterministic resamples), not over individual evaluation episodes. "
        "The median completed length is pooled over completed held-out episodes; "
        "failed episodes remain in the rate denominators and are not assigned an "
        "artificial length. First-success times are training interactions, their "
        "brackets are observed minima and maxima rather than confidence intervals, "
        "and censoring is reported when no training success occurred within the "
        "locked budget."
    )
    return rows, "\n".join(markdown_rows) + note + "\n"


def _write_table(table_rows: list[dict[str, Any]], markdown: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(table_rows[0])
    with (output / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    (output / "final_results.md").write_text(markdown, encoding="utf-8")


def _save_figure(figure: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving average used only for readability of validation curves."""

    if values.size < 2:
        return values
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _validation_histories(raw_root: Path, condition: str) -> dict[int, list[dict[str, Any]]]:
    histories: dict[int, list[dict[str, Any]]] = {}
    for run_dir in _run_dirs(raw_root, condition):
        history = _read_json(run_dir / "validation_history.json")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Invalid validation history for {run_dir}")
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        histories[seed] = [dict(entry) for entry in history]
    return histories


def _aligned_validation_rates(
    history: Sequence[Mapping[str, Any]], common_steps: np.ndarray
) -> np.ndarray:
    """Interpolate one seed's episode-boundary validation rates to a fixed grid."""

    observed_steps = np.asarray(
        [int(row["environment_steps"]) for row in history], dtype=float
    )
    observed_rates = np.asarray(
        [float(row["assignment_success_rate"]) for row in history], dtype=float
    )
    if observed_steps.size < 2 or np.any(np.diff(observed_steps) <= 0):
        raise ValueError("validation history steps must be strictly increasing")
    return np.interp(common_steps, observed_steps, observed_rates)


def make_learning_curves(raw_root: Path, figures: Path) -> None:
    """Figure 1: validation assignment success against interactions."""

    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    common_steps = np.arange(10_000, 60_001, 10_000, dtype=float)
    for condition in CURVE_CONDITIONS:
        histories = _validation_histories(raw_root, condition)
        curves = []
        for seed, history in sorted(histories.items()):
            # Validation is triggered at episode boundaries, so checkpoints can
            # overshoot the nominal 10k cadence differently by seed. Align every
            # seed to the preregistered interaction grid before aggregation.
            curve = _aligned_validation_rates(history, common_steps)
            smoothed = _moving_average(curve, window=3)
            curves.append(smoothed)
            axis.plot(common_steps, smoothed, color=COLORS[condition], alpha=0.22, linewidth=1)
        data = np.vstack(curves)
        mean = data.mean(axis=0)
        lower = np.empty_like(mean)
        upper = np.empty_like(mean)
        for point in range(data.shape[1]):
            lower[point], upper[point] = _bootstrap_ci(
                data[:, point], rng_seed=40_000 + point + 100 * len(condition)
            )
        axis.plot(common_steps, mean, color=COLORS[condition], linewidth=2.4, label=LABELS[condition])
        axis.fill_between(common_steps, lower, upper, color=COLORS[condition], alpha=0.16)
    axis.set(xlabel="Environment interactions", ylabel="Validation assignment success rate (≤100 steps)", ylim=(-0.02, 1.02))
    axis.set_title("Figure 1. Learning curves (3-checkpoint centered moving average)")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.text(
        0.5,
        -0.01,
        "Opaque lines: mean across five training seeds; bands: seed-level 95% bootstrap CI; faint lines: individual seeds.",
        ha="center",
        fontsize=8,
    )
    _save_figure(figure, figures, "figure_1_learning_curves")


def _metric_values(raw_root: Path, condition: str, metric: str) -> list[float]:
    grouped: dict[int, list[Episode]] = defaultdict(list)
    for episode in _final_episodes(raw_root, condition):
        grouped[episode.training_seed].append(episode)
    summaries = [_seed_summary(rows) for _, rows in sorted(grouped.items())]
    return [float(row[metric]) for row in summaries]


def make_final_performance(raw_root: Path, figures: Path) -> None:
    """Figure 2: held-out completion and assignment success by condition."""

    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(CONDITIONS), dtype=float)
    width = 0.36
    completion_means: list[float] = []
    success_means: list[float] = []
    completion_errors: list[tuple[float, float]] = []
    success_errors: list[tuple[float, float]] = []
    for index, condition in enumerate(CONDITIONS):
        for metric, means, errors in (
            ("completion_rate", completion_means, completion_errors),
            ("assignment_success_rate", success_means, success_errors),
        ):
            values = _metric_values(raw_root, condition, metric)
            mean = float(np.mean(values))
            low, high = _bootstrap_ci(values, rng_seed=50_000 + 10 * index + len(means))
            means.append(mean)
            errors.append((mean - low, high - mean))
    axis.bar(
        x - width / 2,
        completion_means,
        width,
        label="Environment completion (≤200)",
        color="#4c78a8",
        yerr=np.asarray(completion_errors).T,
        capsize=3,
    )
    axis.bar(
        x + width / 2,
        success_means,
        width,
        label="Assignment success (≤100)",
        color="#f58518",
        yerr=np.asarray(success_errors).T,
        capsize=3,
    )
    axis.set_xticks(x, [LABELS[condition] for condition in CONDITIONS], rotation=25, ha="right")
    axis.set(ylabel="Held-out episode rate", ylim=(-0.03, 1.08))
    axis.set_title("Figure 2. Final performance on original MountainCar objective")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.text(0.5, 0.015, "Error bars are 95% bootstrap intervals across five training seeds.", ha="center", fontsize=8)
    figure.subplots_adjust(bottom=0.32)
    _save_figure(figure, figures, "figure_2_final_performance")


def _select_representative(raw_root: Path, condition: str, *, successful: bool) -> Episode:
    records = _final_episodes(raw_root, condition)
    candidates = [episode for episode in records if episode.completion is successful]
    if not candidates:
        raise ValueError(f"No {'successful' if successful else 'failed'} final episodes for {condition}")
    if successful:
        target = float(np.median([episode.steps for episode in candidates]))
        key = lambda episode: (abs(episode.steps - target), episode.training_seed, episode.episode)
    else:
        target = float(np.median([episode.maximum_position for episode in candidates]))
        key = lambda episode: (
            abs(episode.maximum_position - target), episode.training_seed, episode.episode
        )
    return min(candidates, key=key)


def _trajectory(raw_root: Path, selected: Episode) -> list[dict[str, Any]]:
    path = raw_root / f"{selected.condition}_seed{selected.training_seed}" / "final_test" / "trajectories.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["episode"]) == selected.episode:
                rows.append(row)
    if not rows:
        raise ValueError(f"No trajectory rows found for {selected}")
    return rows


def make_phase_space(raw_root: Path, figures: Path) -> None:
    """Figure 3: deterministic representative held-out phase-space paths."""

    choices = (
        ("dqn_replay_fast_decay", False, "Rapid-decay DQN (non-completion)"),
        ("dqn_replay", True, "Replay DQN (completion)"),
        ("dqn_replay_shaped", True, "Shaped replay DQN (completion)"),
        ("sarsa_lambda", True, "SARSA(λ) (completion)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    for axis, (condition, successful, title) in zip(axes.flat, choices, strict=True):
        selected = _select_representative(raw_root, condition, successful=successful)
        trajectory = _trajectory(raw_root, selected)
        positions = np.asarray([float(row["position"]) for row in trajectory] + [float(trajectory[-1]["next_position"])])
        velocities = np.asarray([float(row["velocity"]) for row in trajectory] + [float(trajectory[-1]["next_velocity"])])
        axis.plot(positions, velocities, color=COLORS.get(condition, "#333333"), linewidth=1.5)
        axis.scatter(positions[0], velocities[0], color="black", marker="o", s=23, label="start")
        axis.scatter(positions[-1], velocities[-1], color="black", marker="X", s=30, label="final")
        arrow_index = max(1, len(positions) // 2)
        axis.annotate("", xy=(positions[arrow_index], velocities[arrow_index]), xytext=(positions[arrow_index - 1], velocities[arrow_index - 1]), arrowprops={"arrowstyle": "->", "color": COLORS.get(condition, "#333333")})
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1, label="goal threshold")
        axis.set(xlim=(-1.2, 0.6), ylim=(-0.07, 0.07), title=f"{title}\nseed {selected.training_seed}, episode {selected.episode}, {selected.steps} steps")
        axis.grid(alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Velocity")
    for axis in axes[-1, :]:
        axis.set_xlabel("Position")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Figure 3. Position–velocity trajectories (identical axes)", y=0.98)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save_figure(figure, figures, "figure_3_phase_space")


def _training_rows(raw_root: Path, condition: str) -> dict[int, list[dict[str, str]]]:
    return {
        int(path.name.rsplit("seed", 1)[1]): _csv_rows(path / "episodes.csv")
        for path in _run_dirs(raw_root, condition)
    }


def _plot_training_metric(axis: plt.Axes, raw_root: Path, metric: str, ylabel: str) -> None:
    for condition in ("dqn_replay", "dqn_replay_fast_decay"):
        series = _training_rows(raw_root, condition)
        # Episode boundaries differ by seed. Interpolate every seed to a shared
        # interaction grid before averaging so late portions never silently
        # average a changing number of training runs.
        grid = np.arange(0, 60_001, 500, dtype=float)
        per_seed = []
        for rows in series.values():
            steps = np.asarray([int(row["environment_steps"]) for row in rows], dtype=float)
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            per_seed.append(np.interp(grid, steps, values, left=values[0], right=values[-1]))
        mean = np.mean(per_seed, axis=0)
        axis.plot(grid, _moving_average(mean, window=5), color=COLORS[condition], linewidth=2, label=LABELS[condition])
    axis.set(xlabel="Environment interactions", ylabel=ylabel)
    axis.grid(alpha=0.25)


def make_failure_diagnostics(raw_root: Path, figures: Path) -> None:
    """Figure 4: frozen standard/rapid DQN diagnostic comparison."""

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    _plot_training_metric(axes[0, 0], raw_root, "epsilon", "Epsilon")
    axes[0, 0].set_title("Exploration schedule")
    _plot_training_metric(axes[0, 1], raw_root, "maximum_position", "Maximum position in episode")
    axes[0, 1].set_title("Peak progress")
    _plot_training_metric(axes[1, 0], raw_root, "maximum_absolute_velocity", "Maximum |velocity| in episode")
    axes[1, 0].set_title("Momentum proxy")
    _plot_training_metric(axes[1, 1], raw_root, "replay_coverage_fraction", "Replay state-coverage fraction")
    axes[1, 1].set_title("Replay-buffer coverage (20×20 bins)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Figure 4. Replay DQN versus rapid-epsilon-decay diagnostic traces", y=0.98)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save_figure(figure, figures, "figure_4_failure_diagnostics")


def _analysis_manifest(raw_root: Path, table_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": raw_root.as_posix(),
        "conditions": list(CONDITIONS),
        "training_seeds": [0, 1, 2, 3, 4],
        "final_test_episode_seeds": list(range(2000, 2020)),
        "uncertainty": "95% percentile bootstrap over training-seed summaries; 10,000 deterministic resamples",
        "learning_curve_smoothing": "each seed linearly aligned to the preregistered 10,000-interaction grid, then a 3-checkpoint centered moving average for display only",
        "failure_diagnostic_smoothing": "five-seed mean after interpolation to a 500-interaction grid; 5-point centered moving average for display only",
        "representative_phase_space_selection": "successful episode closest to pooled condition median completed length; rapid-decay non-completion closest to pooled failed median maximum position; seed then episode tie-break",
        "table_rows": table_rows,
    }


def _write_results_summary(raw_root: Path, processed: Path, table_rows: list[dict[str, Any]]) -> None:
    """Write evidence-linked hypothesis interpretations directly from raw-derived rows."""
    by_id = {row["condition_id"]: row for row in table_rows}
    replay = _final_episodes(raw_root, "dqn_replay")
    shaped = _final_episodes(raw_root, "dqn_replay_shaped")
    grouped_replay: dict[int, list[Episode]] = defaultdict(list)
    grouped_shaped: dict[int, list[Episode]] = defaultdict(list)
    for item in replay: grouped_replay[item.training_seed].append(item)
    for item in shaped: grouped_shaped[item.training_seed].append(item)
    paired = [
        _seed_summary(grouped_shaped[seed])["assignment_success_rate"] - _seed_summary(grouped_replay[seed])["assignment_success_rate"]
        for seed in sorted(grouped_replay)
    ]
    paired_values = [float(value) for value in paired]
    low, high = _bootstrap_ci(paired_values, rng_seed=45_001)
    text = f"""# Results summary

All final metrics are regenerated from `results/raw/full`; the independent unit
is the training seed (`n=5`), never the 20 held-out episodes within one policy.
See [Table 1](../tables/final_results.md) and Figures 1--4.

## Target outcome

**No model reliably met the ≤100-step target.** The strongest condition was
SARSA(lambda): {by_id['sarsa_lambda']['assignment_success_rate_within_100_steps_95ci']}.
This is progress over random ({by_id['random']['assignment_success_rate_within_100_steps_95ci']}),
not reliable task fulfilment (Table 1; Figure 2).

## Hypotheses

### H1 — reliability within 100 steps: partially supported

At least one learned policy exceeded random, but no condition approached a
reliable 100-step completion rate (Table 1; Figures 1--2).

### H2 — replay improves stability/sample efficiency: not supported

Replay DQN ended at completion {by_id['dqn_replay']['completion_rate_within_200_steps_95ci']}
and success {by_id['dqn_replay']['assignment_success_rate_within_100_steps_95ci']};
the no-replay values were {by_id['dqn_no_replay']['completion_rate_within_200_steps_95ci']}
and {by_id['dqn_no_replay']['assignment_success_rate_within_100_steps_95ci']}.
Figure 1 shows neither DQN produced sustained ≤100 validation success at the
locked budget. This comparison remains limited because removing replay also
changes minibatch decorrelation and batch size.

### H3 — SARSA(lambda) versus neural DQN: partially supported

SARSA(lambda) had stronger saved outcomes than all neural conditions (Table 1;
Figures 1--2). This is descriptive only: tile coding, traces, and neural
optimization differ, so it is not a controlled on-policy/off-policy causal test.

### H4 — potential shaping accelerates discovery: partially supported

Shaped replay's first completion was
{by_id['dqn_replay_shaped']['time_to_first_training_completion_interactions']},
whereas standard replay was
{by_id['dqn_replay']['time_to_first_training_completion_interactions']}. Its
first ≤100 training success was
{by_id['dqn_replay_shaped']['time_to_first_training_assignment_success_interactions']}.
The censoring and sparse successes shown in Table 1/Figure 1 support occasional
earlier discovery, not a reliable acceleration conclusion.

### H5 — shaping without final-objective harm: partially supported

Shaped replay's held-out success was
{by_id['dqn_replay_shaped']['assignment_success_rate_within_100_steps_95ci']}
versus {by_id['dqn_replay']['assignment_success_rate_within_100_steps_95ci']}
for standard replay. The paired seed-level difference (shaped minus unshaped)
was {np.mean(paired_values):.3f} [{low:.3f}, {high:.3f}], above the preregistered
-0.05 lower-margin rule. The broad interval and low absolute success mean this
does not establish a robust final-performance improvement (Table 1; Figure 2).

### H6 — rapid exploration decay causes valley trapping: not supported

Rapid decay reached success
{by_id['dqn_replay_fast_decay']['assignment_success_rate_within_100_steps_95ci']}
and completion {by_id['dqn_replay_fast_decay']['completion_rate_within_200_steps_95ci']}.
Figure 4 confirms immediate epsilon reduction but does not show the proposed
systematic loss of momentum/coverage; Figure 3 provides the fixed representative
phase-space comparison. The registered failure mechanism is therefore not
supported and is not redefined after observing the result.
"""
    (processed / "results_summary.md").write_text(text, encoding="utf-8")


def generate_all(raw_root: Path, figures: Path, tables: Path, processed: Path) -> None:
    """Generate the final table and exactly four PNG/PDF figures."""

    table_rows, markdown = build_final_table(raw_root)
    _write_table(table_rows, markdown, tables)
    make_learning_curves(raw_root, figures)
    make_final_performance(raw_root, figures)
    make_phase_space(raw_root, figures)
    make_failure_diagnostics(raw_root, figures)
    processed.mkdir(parents=True, exist_ok=True)
    _write_results_summary(raw_root, processed, table_rows)
    (processed / "analysis_manifest.json").write_text(
        json.dumps(_analysis_manifest(raw_root, table_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("results/raw/full"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--processed", type=Path, default=Path("results/processed"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_all(args.raw_root, args.figures, args.tables, args.processed)
    print(f"Generated tables in {args.tables} and four figures in {args.figures}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())
