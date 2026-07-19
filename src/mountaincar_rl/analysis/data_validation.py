"""Validate immutable final-suite artifacts before statistical analysis.

The validator intentionally reads only saved artifacts.  It never recreates an
episode, modifies a checkpoint, or writes into ``results/raw``.  Checks operate
at the run and training-seed levels so a missing seed cannot be hidden by the
larger number of within-policy evaluation episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mountaincar_rl.experiments.run_suite import SEEDS, TEMPLATES, _config_for_seed
from mountaincar_rl.utils.checkpoints import config_fingerprint, load_agent_checkpoint
from mountaincar_rl.utils.config import load_config


EXPECTED_CONDITIONS = tuple(TEMPLATES)
FINAL_TEST_SEEDS = tuple(range(2000, 2020))
VALIDATION_SEEDS = tuple(range(1000, 1020))
MAX_EPISODE_STEPS = 200


@dataclass(frozen=True)
class Check:
    """One transparent data-quality assertion for the Markdown report."""

    name: str
    status: str
    details: str


def _json(path: Path) -> Any:
    """Read a JSON artifact while leaving a path-specific failure visible."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact, rejecting malformed headers/records."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV lacks a header: {path}")
            rows = list(reader)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing artifact: {path}") from error
    if any(None in row for row in rows):
        raise ValueError(f"CSV row has extra unnamed fields: {path}")
    return rows


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected boolean text, got {value!r}")


def _finite_numbers(value: Any, path: str = "root") -> list[str]:
    """Return locations of non-finite values in a parsed JSON-like artifact."""

    failures: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            failures.extend(_finite_numbers(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failures.extend(_finite_numbers(nested, f"{path}[{index}]"))
    return failures


def _finite_csv_values(rows: Iterable[Mapping[str, str]], path: Path) -> list[str]:
    """Detect serialized NaN/infinity in every numeric-looking CSV value."""

    failures: list[str] = []
    for row_index, row in enumerate(rows):
        for key, text in row.items():
            if text is None or text == "":
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if not math.isfinite(value):
                failures.append(f"{path}:{row_index + 2}:{key}")
    return failures


def _status(failures: Sequence[str], *, warnings: Sequence[str] = ()) -> str:
    if failures:
        return "FAIL"
    if warnings:
        return "WARNING"
    return "PASS"


def _checkpoints(
    root: Path,
    condition: str,
    seed: int,
    config: Mapping[str, Any],
) -> list[str]:
    """Validate every required learned-agent checkpoint envelope."""

    if config["agent"]["type"] == "random":
        return []
    problems: list[str] = []
    checkpoint_dir = root / "checkpoints" / "full" / f"{condition}_seed{seed}"
    for kind in ("latest", "last", "best", "final"):
        path = checkpoint_dir / f"{kind}.pt"
        try:
            payload = load_agent_checkpoint(path, expected_config=config)
            if not isinstance(payload.get("training_state"), Mapping):
                problems.append(f"{path}: missing training_state")
        except Exception as error:  # report a corrupt artifact; do not hide it
            problems.append(f"{path}: {type(error).__name__}: {error}")
    return problems


def validate_final_results(repository_root: Path) -> list[Check]:
    """Return all final-suite checks based on existing raw and locked artifacts."""

    root = repository_root.resolve()
    raw_root = root / "results" / "raw" / "full"
    locked = load_config(root / "configs" / "full" / "locked_manifest.yaml")
    final_seed_config = load_config(root / "configs" / "full" / "final_test.yaml")
    final_seed_list = tuple(int(seed) for seed in final_seed_config["episode_seeds"])
    checks: list[Check] = []

    conditions = locked.get("conditions", {})
    missing_conditions = sorted(set(EXPECTED_CONDITIONS) - set(conditions))
    unexpected_conditions = sorted(set(conditions) - set(EXPECTED_CONDITIONS))
    checks.append(Check(
        "Required experimental conditions",
        _status(missing_conditions, warnings=unexpected_conditions),
        "all six preregistered conditions are present"
        if not missing_conditions and not unexpected_conditions
        else f"missing={missing_conditions}; unexpected={unexpected_conditions}",
    ))
    checks.append(Check(
        "Final-test seed registry",
        _status([] if final_seed_list == FINAL_TEST_SEEDS else ["mismatch"]),
        f"configured seeds={list(final_seed_list)}; expected={list(FINAL_TEST_SEEDS)}",
    ))

    per_run_final_counts: dict[tuple[str, int], int] = {}
    config_problems: list[str] = []
    config_warnings: list[str] = []
    missing_runs: list[str] = []
    duplicate_runs: list[str] = []
    numeric_problems: list[str] = []
    episode_problems: list[str] = []
    seed_overlap_problems: list[str] = []
    reward_problems: list[str] = []
    evaluation_count_problems: list[str] = []
    checkpoint_problems: list[str] = []
    observed_dirs: defaultdict[tuple[str, int], list[Path]] = defaultdict(list)

    for directory in raw_root.glob("*_seed*"):
        if not directory.is_dir():
            continue
        snapshot_path = directory / "config.snapshot.json"
        if not snapshot_path.is_file():
            continue
        try:
            snapshot = _json(snapshot_path)
            experiment = snapshot["experiment"]
            observed_dirs[(str(experiment["condition"]), int(experiment["training_seed"]))].append(directory)
        except (KeyError, TypeError, ValueError) as error:
            config_problems.append(f"{directory}: malformed snapshot ({error})")

    for condition in EXPECTED_CONDITIONS:
        template_path = root / TEMPLATES[condition]
        template = load_config(template_path)
        locked_row = conditions.get(condition, {})
        locked_hash = locked_row.get("resolved_config_hash") if isinstance(locked_row, Mapping) else None
        actual_locked_hash = config_fingerprint(template)
        if locked_hash != actual_locked_hash:
            config_problems.append(
                f"{condition}: locked hash {locked_hash!r} != current template hash {actual_locked_hash}"
            )
        for seed in SEEDS:
            run_key = (condition, seed)
            run_dirs = observed_dirs.get(run_key, [])
            if not run_dirs:
                missing_runs.append(f"{condition}/seed{seed}")
                continue
            if len(run_dirs) != 1:
                duplicate_runs.append(f"{condition}/seed{seed}: {[str(path) for path in run_dirs]}")
                continue
            run_dir = run_dirs[0]
            expected_config = _config_for_seed(template, condition, seed)
            expected_hash = config_fingerprint(expected_config)
            required = [
                run_dir / "config.snapshot.json", run_dir / "summary.json", run_dir / "episodes.csv",
                run_dir / "final_test" / "summary.json", run_dir / "final_test" / "episodes.csv",
                run_dir / "final_test" / "trajectories.jsonl",
            ]
            missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
            if missing:
                missing_runs.extend(missing)
                continue
            snapshot = _json(run_dir / "config.snapshot.json")
            if snapshot != expected_config:
                config_problems.append(f"{run_dir}: snapshot differs from locked per-seed configuration")
            if config_fingerprint(snapshot) != expected_hash:
                config_problems.append(f"{run_dir}: snapshot fingerprint mismatch")
            summary = _json(run_dir / "summary.json")
            final_summary = _json(run_dir / "final_test" / "summary.json")
            if summary.get("status", "complete") != "complete":
                missing_runs.append(f"{run_dir}: training status is {summary.get('status')!r}")
            for artifact, value in (("training summary", summary), ("final summary", final_summary)):
                if condition == "random" and artifact == "training summary" and "config_hash" not in value:
                    config_warnings.append(
                        f"{run_dir.name}: random training summary predates config_hash logging; "
                        "snapshot and held-out evaluation hash were checked instead"
                    )
                    continue
                if value.get("config_hash") != expected_hash:
                    config_problems.append(f"{run_dir}: {artifact} config_hash mismatch")
            if tuple(int(item) for item in final_summary.get("evaluation_episode_seeds", [])) != FINAL_TEST_SEEDS:
                seed_overlap_problems.append(f"{run_dir}: final summary does not contain exactly held-out seeds")
            validation_seeds = tuple(int(item) for item in snapshot["evaluation"]["validation_episode_seeds"])
            if set(validation_seeds) != set(VALIDATION_SEEDS):
                seed_overlap_problems.append(f"{run_dir}: validation seed set differs from 1000--1019")
            if set(validation_seeds) & set(FINAL_TEST_SEEDS):
                seed_overlap_problems.append(f"{run_dir}: validation/final-test seed overlap")
            if set(SEEDS) & set(FINAL_TEST_SEEDS):
                seed_overlap_problems.append("fixed training/final-test seed overlap")

            # The non-learning random baseline has no periodic checkpoint
            # validation, so it legitimately lacks validation_episodes.csv.
            all_csvs = [run_dir / "episodes.csv", run_dir / "final_test" / "episodes.csv"]
            validation_csv = run_dir / "validation_episodes.csv"
            if validation_csv.is_file():
                all_csvs.insert(1, validation_csv)
            for path in all_csvs:
                rows = _csv(path)
                numeric_problems.extend(_finite_csv_values(rows, path))
                for row_index, row in enumerate(rows, start=2):
                    try:
                        steps = int(row["steps"])
                        complete = _as_bool(row["environment_completion"])
                        success = _as_bool(row["assignment_success"])
                    except (KeyError, ValueError) as error:
                        episode_problems.append(f"{path}:{row_index}: {error}")
                        continue
                    if not 1 <= steps <= MAX_EPISODE_STEPS:
                        episode_problems.append(f"{path}:{row_index}: impossible steps={steps}")
                    if success and (steps > 100 or not complete):
                        episode_problems.append(f"{path}:{row_index}: invalid assignment success at {steps} steps")
                    if complete and steps <= 100 and not success:
                        episode_problems.append(f"{path}:{row_index}: completion within 100 not marked assignment success")
                    if path.parent.name == "final_test":
                        try:
                            if float(row["episode_return"]) != float(row["original_return"]):
                                reward_problems.append(f"{path}:{row_index}: final episode_return != original_return")
                        except (KeyError, ValueError):
                            reward_problems.append(f"{path}:{row_index}: missing final original reward")
                        if row.get("shaped_return", "") or row.get("shaping_component", ""):
                            reward_problems.append(f"{path}:{row_index}: final evaluation contains shaped reward")
                        if row.get("config_hash") != expected_hash:
                            config_problems.append(f"{path}:{row_index}: final row config_hash mismatch")
                if path.parent.name == "final_test":
                    final_rows = rows
                    per_run_final_counts[run_key] = len(final_rows)
                    seeds = [int(row["environment_seed"]) for row in final_rows]
                    if seeds != list(FINAL_TEST_SEEDS) or len(set(seeds)) != len(seeds):
                        evaluation_count_problems.append(f"{path}: final-test episode seeds are missing/duplicated")
                    if len(final_rows) != int(final_summary.get("episodes", -1)):
                        evaluation_count_problems.append(f"{path}: CSV/summary episode count mismatch")

            # Parse every saved structured result file, including policy and
            # state metadata not otherwise needed for the episode-level checks.
            for json_path in sorted(run_dir.rglob("*.json")):
                numeric_problems.extend(f"{json_path}:{where}" for where in _finite_numbers(_json(json_path)))
            # Streaming JSONL validation makes corrupt trajectory data visible without retaining it.
            for trajectory_path in sorted(run_dir.rglob("*.jsonl")):
                with trajectory_path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            numeric_problems.append(f"{trajectory_path}:{line_number}: invalid JSON ({error})")
                            continue
                        numeric_problems.extend(f"{trajectory_path}:{line_number}:{where}" for where in _finite_numbers(record))
            checkpoint_problems.extend(_checkpoints(root, condition, seed, expected_config))

    extra_runs = sorted(set(observed_dirs) - {(condition, seed) for condition in EXPECTED_CONDITIONS for seed in SEEDS})
    if extra_runs:
        duplicate_runs.extend(f"unexpected run directory {condition}/seed{seed}" for condition, seed in extra_runs)
    checks.append(Check("Planned training seeds and required raw artifacts", _status(missing_runs),
                        "30/30 condition-seed runs present and complete" if not missing_runs else "; ".join(missing_runs)))
    checks.append(Check("Duplicate or unexpected final runs", _status(duplicate_runs),
                        "exactly one raw run per condition and training seed" if not duplicate_runs else "; ".join(duplicate_runs)))
    checks.append(Check("Locked configuration hashes and per-seed snapshots", _status(config_problems, warnings=config_warnings),
                        "locked templates and all run-specific fingerprints match" if not config_problems and not config_warnings
                        else "; ".join(config_problems[:8] if config_problems else config_warnings[:5])))
    checks.append(Check("Finite numeric values", _status(numeric_problems),
                        "all parsed CSV, JSON, and JSONL numeric values are finite" if not numeric_problems else "; ".join(numeric_problems[:8])))
    checks.append(Check("Episode lengths and <=100 success semantics", _status(episode_problems),
                        "all episodes have 1--200 steps and valid completion/success flags" if not episode_problems else "; ".join(episode_problems[:8])))
    checks.append(Check("Disjoint training, validation, and held-out seeds", _status(seed_overlap_problems),
                        "training=0--4, validation=1000--1019, final test=2000--2019" if not seed_overlap_problems else "; ".join(seed_overlap_problems[:8])))
    checks.append(Check("Original-objective final evaluation", _status(reward_problems),
                        "all final-test episode_return values equal original_return; shaped fields are empty" if not reward_problems else "; ".join(reward_problems[:8])))
    expected_count = len(FINAL_TEST_SEEDS)
    unequal = [f"{condition}/seed{seed}={count}" for (condition, seed), count in sorted(per_run_final_counts.items()) if count != expected_count]
    evaluation_count_problems.extend(unequal)
    checks.append(Check("Equal final-test episode counts", _status(evaluation_count_problems),
                        "each of 30 policies has 20 held-out episodes" if not evaluation_count_problems else "; ".join(evaluation_count_problems[:8])))
    checks.append(Check("Checkpoint completeness", _status(checkpoint_problems),
                        "all 100 learned-agent checkpoint envelopes load and match their run configuration" if not checkpoint_problems else "; ".join(checkpoint_problems[:4])))
    return checks


def render_report(checks: Sequence[Check]) -> str:
    """Render a concise, self-contained Markdown validation report."""

    failure_count = sum(check.status == "FAIL" for check in checks)
    warning_count = sum(check.status == "WARNING" for check in checks)
    lines = [
        "# Final Result Data Validation",
        "",
        "Generated only from existing `results/raw/full` artifacts and locked configurations; no training or evaluation was rerun.",
        "",
        f"Overall: **{'FAIL' if failure_count else 'PASS'}** — {len(checks) - failure_count - warning_count} pass, {warning_count} warning, {failure_count} failure.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for check in checks:
        details = check.details.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check.name} | {check.status} | {details} |")
    lines.extend([
        "",
        "Interpretation: a `WARNING` flags a non-fatal audit concern; a `FAIL` means the affected result must not be treated as validated until the saved artifact is repaired or its exclusion is explicitly documented. Confidence intervals and final comparisons must use training seeds, not the 20 within-policy held-out episodes as independent runs.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/processed/data_validation_report.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = validate_final_results(args.repository_root)
    output = args.output if args.output.is_absolute() else args.repository_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(checks), encoding="utf-8")
    print(f"wrote {output}")
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Check", "render_report", "validate_final_results"]
