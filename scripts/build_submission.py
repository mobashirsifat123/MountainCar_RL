"""Build the minimal submission bundle from verified repository artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import gymnasium
import matplotlib
import numpy
import torch
import yaml

from mountaincar_rl.utils.checkpoints import config_fingerprint
from mountaincar_rl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
LEARNED_CONDITIONS = (
    "sarsa_lambda",
    "dqn_no_replay",
    "dqn_replay",
    "dqn_replay_shaped",
    "dqn_replay_fast_decay",
)
ESSENTIAL_FIGURES = {
    "figure_1_learning_curves": (
        "Assignment-success learning curves across training seeds",
        "Results: learning and sample efficiency",
    ),
    "figure_2_final_performance": (
        "Held-out completion and at-most-100-step success comparison",
        "Results: held-out performance",
    ),
    "figure_3_phase_space": (
        "Representative position-velocity trajectories",
        "Behavioral and failure analysis",
    ),
    "figure_4_failure_diagnostics": (
        "Replay DQN versus rapid-epsilon-decay training diagnostics",
        "Behavioral and failure analysis",
    ),
}
PDF_NAME = "MountainCar_RL_Recruitment_Report.pdf"
VOLATILE_UNCHECKSUMMED = {
    "verification_output.txt",
    "archive_verification.md",
}


def _copy(source: Path, destination: Path) -> None:
    """Copy one required file, failing clearly when its source is absent."""

    if not source.is_file():
        raise FileNotFoundError(f"Required submission source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_final_rows() -> list[dict[str, str]]:
    """Read the machine-generated final table used for the submission summary."""

    path = ROOT / "results/tables/final_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError(f"Expected six final-result rows, found {len(rows)}")
    return rows


def _final_results_markdown(rows: list[dict[str, str]]) -> str:
    """Render the concise submission summary only from generated table cells."""

    by_id = {row["condition_id"]: row for row in rows}
    sarsa = by_id["sarsa_lambda"]
    online = by_id["dqn_no_replay"]
    replay = by_id["dqn_replay"]
    shaped = by_id["dqn_replay_shaped"]
    rapid = by_id["dqn_replay_fast_decay"]
    checkpoint_lines = "\n".join(
        f"- `best_models/sarsa_lambda_seed{seed}/best.pt` "
        f"(repository source: `checkpoints/full/sarsa_lambda_seed{seed}/best.pt`)"
        for seed in range(5)
    )
    return f"""# Final result summary

## Task and success definition

The task is Gymnasium `MountainCar-v0`. Environment completion means reaching
the flag within the normal 200-step limit. Assignment success is stricter:
reaching the flag in at most 100 actual environment steps. A completion in
101--200 steps is not assignment success.

## Best measured condition

The strongest aggregate condition was **{sarsa['condition']}**:

- training seeds: {sarsa['training_seeds']};
- held-out episodes per trained policy: {sarsa['evaluation_episodes_per_seed']};
- environment completion within 200 steps: {sarsa['completion_rate_within_200_steps_95ci']};
- assignment success within 100 steps: {sarsa['assignment_success_rate_within_100_steps_95ci']};
- median length among completed held-out episodes: {sarsa['median_steps_among_completed_episodes']} steps;
- mean original environment return: {sarsa['mean_original_environment_return_95ci']}.

Intervals are 95% percentile bootstrap intervals over the five training-seed
summaries, not over pooled evaluation episodes. The completed-episode median is
conditional and pooled; failed episodes remain in the rate denominators.

## Evaluation protocol

Training seeds were 0--4. Checkpoints were selected only with validation episode
seeds 1000--1019. Greedy held-out evaluation then used the same 20 disjoint seeds
2000--2019 for every trained policy and always used the original MountainCar
reward, including for the shaped agent.

## Main findings

- **Replay:** Online DQN completed
  {online['completion_rate_within_200_steps_95ci']} of held-out episodes versus
  {replay['completion_rate_within_200_steps_95ci']} for replay DQN; both had
  {replay['assignment_success_rate_within_100_steps_95ci']} assignment success.
  Replay did not establish a stability or sample-efficiency advantage at the
  locked 60,000-interaction budget. Removing replay also changed batch size and
  temporal decorrelation.
- **Shaping:** Shaped replay completed
  {shaped['completion_rate_within_200_steps_95ci']} and achieved
  {shaped['assignment_success_rate_within_100_steps_95ci']} assignment success.
  Its first training completion was
  {shaped['time_to_first_training_completion_interactions']}; sparse successes
  and broad seed intervals do not establish a robust final-performance benefit.
- **Rapid-decay intervention:** Rapid-epsilon-decay DQN completed
  {rapid['completion_rate_within_200_steps_95ci']} and achieved
  {rapid['assignment_success_rate_within_100_steps_95ci']} assignment success.
  Aggregate diagnostics did not support the preregistered claim that rapid decay
  systematically trapped the agent in low-momentum valley trajectories.

No condition reliably met the at-most-100-step assignment target.

## Validation-selected best checkpoints

There is no checkpoint selected across training seeds using held-out results.
The five validation-selected checkpoints for the strongest condition are:

{checkpoint_lines}

The exact submitted seed-0 best-checkpoint path is
`best_models/sarsa_lambda_seed0/best.pt`. Best checkpoints for every other
learned condition and training seed are also included under `best_models/`.

## Limitations

The study uses five training seeds, a simple deterministic benchmark, bounded
validation tuning, and a practical 60,000-interaction budget. SARSA and DQN use
different function approximators, so their comparison is not a controlled
on-policy/off-policy causal test. The shaping potential is an interpretable
energy proxy rather than exact mechanical energy. Conclusions may not
generalize to larger or stochastic reinforcement-learning tasks.
"""


def _submission_readme() -> str:
    """Return the submission-bundle cover page without duplicating the report."""

    return """# MountainCar RL recruitment submission

This is the professor-facing package for **Learning to Build
Momentum: Experience Replay, On-Policy Learning, and Reward Shaping in
MountainCar**.

The assignment target is to reach the `MountainCar-v0` flag within 100 steps.
The strongest measured condition was tile-coded SARSA(lambda): five training
seeds, 20 held-out episodes per seed, 1.000 [1.000, 1.000] completion within
200 steps and 0.380 [0.330, 0.420] success within 100 steps. The target was not
met reliably.

Compared methods: random policy, SARSA(lambda), DQN without replay, DQN with
uniform replay, replay DQN with potential-based energy-deficit shaping, and a
rapid-epsilon-decay replay-DQN intervention.

Repository URL: not configured in the delivered local repository copy.

This ZIP is a professor-facing evidence bundle, not an installable source tree.
Run installation and reproduction commands from a repository checkout; the
required source files are listed in `source_manifest.txt`.

## Start here

- `MountainCar_RL_Recruitment_Report.pdf` for the final paper;
- `report.md` for its Markdown source;
- `final_results.md` for the verified outcome summary;
- `tables/` and `figures/` for generated result artifacts;
- `best_models/` for every validation-selected per-seed best checkpoint;
- `configs/` for resolved locked final configurations;
- `reproduction_commands.txt` for exact commands, run from the repository root;
- `source_manifest.txt` for the source files supporting this bundle;
- `checksum_manifest.sha256` for integrity verification.

## Commands

Install from the repository root:

```bash
uv sync --frozen --python 3.12 --extra dev
```

Quickly evaluate the strongest seed-0 checkpoint on two validation episodes:

```bash
uv run --python 3.12 mountaincar-evaluate \
  --config submission/best_models/sarsa_lambda_seed0/config.snapshot.json \
  --split validation --episodes 2 \
  --checkpoint submission/best_models/sarsa_lambda_seed0/best.pt \
  --output-dir results/reproduction/sarsa_seed0_validation \
  --capture-trajectories
```

The full expensive reproduction command is
`bash scripts/run_full_experiments.sh`. Run `make verify-submission` for the
fast release check.

No model was retrained and no result was changed while creating this package.
"""


def _reproduction_commands() -> str:
    """Return exact repository-root commands without triggering full training."""

    return """# Run every command from the repository root.

# 1. Install the locked Python 3.12 environment.
uv sync --frozen --python 3.12 --extra dev

# 2. Run the complete unit and integration test suite.
uv run --python 3.12 pytest

# 3. Run short train/evaluate smoke checks for all six conditions.
bash scripts/smoke_test.sh

# 4. Train one short selected SARSA model (overwrites smoke artifacts only).
uv run --python 3.12 mountaincar-train --config configs/smoke/sarsa_lambda.yaml --overwrite

# 5. Evaluate a submitted SARSA checkpoint on all fixed held-out seeds.
uv run --python 3.12 mountaincar-evaluate --config submission/best_models/sarsa_lambda_seed0/config.snapshot.json --split final_test --seeds-config submission/configs/final_test.yaml --checkpoint submission/best_models/sarsa_lambda_seed0/best.pt --output-dir results/reproduction/sarsa_lambda_seed0_final_test --capture-trajectories

# 6. Reproduce the complete locked suite (expensive; all seeds/conditions).
bash scripts/run_full_experiments.sh

# 7. Regenerate tables and figures from existing full raw results; no training.
uv run --python 3.12 python -m mountaincar_rl.analysis.make_all_figures

# 8. Verify the finished submission bundle.
make verify-submission
"""


def _source_manifest() -> str:
    """List important implementation, configuration, test, and provenance files."""

    fixed = (
        "Makefile",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "docs/algorithm_notes.md",
        "experiments/experiment_log.md",
        "experiments/hypotheses.md",
        "report/report.md",
        "report/report.css",
    )
    paths = {Path(item) for item in fixed}
    for pattern in (
        "src/mountaincar_rl/**/*.py",
        "tests/test_*.py",
        "scripts/*.py",
        "scripts/*.sh",
        "configs/**/*.yaml",
        "configs/**/*.md",
    ):
        paths.update(path.relative_to(ROOT) for path in ROOT.glob(pattern))
    existing = sorted(path.as_posix() for path in paths if (ROOT / path).is_file())
    return "\n".join(existing) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_text() -> str:
    """Record study and release-verification environments without private paths."""

    study = json.loads(
        (ROOT / "results/raw/full/sarsa_lambda_seed0/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    study_packages = study["packages"]
    return f"""MountainCar RL environment record

Saved full-experiment environment:
  operating_system: {study['platform']['system']} {study['platform']['release']} ({study['platform']['machine']})
  python: {study['python']['version']} ({study['python']['implementation']})
  device: {study['device']}
  gymnasium: {study_packages['gymnasium']}
  matplotlib: {study_packages['matplotlib']}
  numpy: {study_packages['numpy']}
  torch: {study_packages['torch']}
  pyyaml: {study_packages['pyyaml']}

Final local verification environment:
  operating_system: {platform.system()} {platform.release()} ({platform.machine()})
  python: {platform.python_version()} ({platform.python_implementation()})
  gymnasium: {gymnasium.__version__}
  matplotlib: {matplotlib.__version__}
  numpy: {numpy.__version__}
  torch: {torch.__version__}
  pyyaml: {yaml.__version__}

Installation command:
  uv sync --frozen --python 3.12 --extra dev
"""


def _figure_manifest() -> str:
    lines = [
        "# Figure manifest",
        "",
        "All figures are generated from `results/raw/full/` with:",
        "",
        "```bash",
        "uv run --python 3.12 python -m mountaincar_rl.analysis.make_all_figures",
        "```",
        "",
        "| File | Purpose | Source data | Report section | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for stem, (purpose, section) in ESSENTIAL_FIGURES.items():
        for suffix in (".png", ".pdf"):
            path = SUBMISSION / "figures" / f"{stem}{suffix}"
            lines.append(
                f"| `{path.name}` | {purpose} | `results/raw/full/` | "
                f"{section} | `{_sha256(path)}` |"
            )
    return "\n".join(lines) + "\n"


def _copy_locked_configs() -> None:
    manifest_path = ROOT / "configs/full/locked_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    _copy(manifest_path, SUBMISSION / "configs/locked_manifest.yaml")
    _copy(ROOT / "configs/full/final_test.yaml", SUBMISSION / "configs/final_test.yaml")
    for condition, record in manifest["conditions"].items():
        source = ROOT / "configs" / record["config"]
        resolved = load_config(source)
        actual_hash = config_fingerprint(resolved)
        if actual_hash != record["resolved_config_hash"]:
            raise ValueError(f"Locked configuration hash mismatch for {condition}")
        destination = SUBMISSION / "configs" / f"{condition}.yaml"
        destination.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        if config_fingerprint(load_config(destination)) != actual_hash:
            raise ValueError(f"Submitted resolved config changed hash for {condition}")


def _model_metadata(condition: str, seed: int, checkpoint: Path) -> dict[str, object]:
    run_dir = ROOT / f"results/raw/full/{condition}_seed{seed}"
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final = json.loads(
        (run_dir / "final_test/summary.json").read_text(encoding="utf-8")
    )
    config_relative = f"best_models/{condition}_seed{seed}/config.snapshot.json"
    checkpoint_relative = f"best_models/{condition}_seed{seed}/best.pt"
    return {
        "schema_version": 1,
        "algorithm": training["agent_type"],
        "condition": condition,
        "training_seed": seed,
        "checkpoint_type": "best_validation",
        "checkpoint": checkpoint_relative,
        "checkpoint_sha256": _sha256(checkpoint),
        "configuration": config_relative,
        "configuration_hash": training["config_hash"],
        "validation_metric": training["best_validation"],
        "final_evaluation_metric": {
            "episodes": final["episodes"],
            "episode_seeds": final["evaluation_episode_seeds"],
            "environment_completion_rate": final["environment_completion_rate"],
            "assignment_success_rate": final["assignment_success_rate"],
            "mean_original_environment_return": final["mean_episode_return"],
            "median_steps_among_completions": final[
                "median_steps_among_completions"
            ],
            "objective": final["objective"],
        },
        "loading_command": (
            "uv run --python 3.12 mountaincar-evaluate "
            f"--config submission/{config_relative} --split validation --episodes 2 "
            f"--checkpoint submission/{checkpoint_relative} "
            f"--output-dir results/reproduction/{condition}_seed{seed}_validation "
            "--capture-trajectories"
        ),
    }


def _write_checksums() -> None:
    """Hash every submitted regular file except the checksum manifest itself."""

    manifest = SUBMISSION / "checksum_manifest.sha256"
    files = sorted(
        path
        for path in SUBMISSION.rglob("*")
        if path.is_file()
        and path != manifest
        and path.relative_to(SUBMISSION).as_posix() not in VOLATILE_UNCHECKSUMMED
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(SUBMISSION).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_submission() -> None:
    """Create or deterministically refresh the essential submission artifacts."""

    if SUBMISSION.exists():
        shutil.rmtree(SUBMISSION)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    for directory in ("figures", "tables", "best_models", "configs"):
        (SUBMISSION / directory).mkdir(parents=True, exist_ok=True)

    (SUBMISSION / "README.md").write_text(_submission_readme(), encoding="utf-8")
    report = (ROOT / "report/report.md").read_text(encoding="utf-8")
    report = report.replace("../results/figures/", "figures/")
    report = report.replace("../results/tables/", "tables/")
    (SUBMISSION / "report.md").write_text(report, encoding="utf-8")
    rows = _read_final_rows()
    (SUBMISSION / "final_results.md").write_text(
        _final_results_markdown(rows), encoding="utf-8"
    )
    _copy(ROOT / "requirements.txt", SUBMISSION / "requirements.txt")
    (SUBMISSION / "reproduction_commands.txt").write_text(
        _reproduction_commands(), encoding="utf-8"
    )
    (SUBMISSION / "environment.txt").write_text(
        _environment_text(), encoding="utf-8"
    )
    _copy(ROOT / "output/pdf" / PDF_NAME, SUBMISSION / PDF_NAME)
    _copy(
        ROOT / "output/pdf/report_build_log.txt",
        SUBMISSION / "report_build_log.txt",
    )

    for stem in ESSENTIAL_FIGURES:
        for suffix in (".png", ".pdf"):
            name = f"{stem}{suffix}"
            _copy(ROOT / "results/figures" / name, SUBMISSION / "figures" / name)
    for name in ("final_results.csv", "final_results.md"):
        _copy(ROOT / "results/tables" / name, SUBMISSION / "tables" / name)
    _copy_locked_configs()
    for condition in LEARNED_CONDITIONS:
        for seed in range(5):
            source = ROOT / f"checkpoints/full/{condition}_seed{seed}/best.pt"
            destination = SUBMISSION / f"best_models/{condition}_seed{seed}/best.pt"
            _copy(source, destination)
            _copy(
                ROOT / f"results/raw/full/{condition}_seed{seed}/config.snapshot.json",
                SUBMISSION
                / f"best_models/{condition}_seed{seed}/config.snapshot.json",
            )
            metadata = _model_metadata(condition, seed, destination)
            (destination.parent / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    (SUBMISSION / "figure_manifest.md").write_text(
        _figure_manifest(), encoding="utf-8"
    )
    (SUBMISSION / "source_manifest.txt").write_text(
        _source_manifest(), encoding="utf-8"
    )
    _write_checksums()

    file_count = sum(path.is_file() for path in SUBMISSION.rglob("*"))
    print(f"Built {SUBMISSION.relative_to(ROOT)} with {file_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.checksums_only:
        _write_checksums()
        print("Updated submission/checksum_manifest.sha256")
    else:
        build_submission()
