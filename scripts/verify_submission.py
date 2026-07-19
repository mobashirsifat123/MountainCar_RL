"""Fast integrity and scientific-artifact checks for the submission bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.experiments.agent_factory import build_agent
from mountaincar_rl.experiments.evaluate import run_evaluation
from mountaincar_rl.environments.factory import make_env
from mountaincar_rl.utils.checkpoints import (
    config_fingerprint,
    load_agent_checkpoint,
)
from mountaincar_rl.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
CONDITION_IDS = {
    "random",
    "sarsa_lambda",
    "dqn_no_replay",
    "dqn_replay",
    "dqn_replay_shaped",
    "dqn_replay_fast_decay",
}
REQUIRED_TOP_LEVEL = {
    "README.md",
    "MountainCar_RL_Recruitment_Report.pdf",
    "report.md",
    "final_results.md",
    "requirements.txt",
    "reproduction_commands.txt",
    "source_manifest.txt",
    "figure_manifest.md",
    "environment.txt",
    "report_build_log.txt",
    "checksum_manifest.sha256",
}
VOLATILE_UNCHECKSUMMED = {
    Path("verification_output.txt"),
    Path("archive_verification.md"),
}


def _verify_required_files() -> None:
    """Require every deliverable and reject cache/temporary artifacts."""

    missing = sorted(name for name in REQUIRED_TOP_LEVEL if not (SUBMISSION / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing top-level submission files: {missing}")
    for directory in ("figures", "tables", "best_models", "configs"):
        if not (SUBMISSION / directory).is_dir():
            raise FileNotFoundError(f"Missing submission directory: {directory}")
    if len(list((SUBMISSION / "figures").glob("*.png"))) != 4:
        raise ValueError("Submission must contain exactly four PNG figures")
    if len(list((SUBMISSION / "figures").glob("*.pdf"))) != 4:
        raise ValueError("Submission must contain exactly four PDF figures")
    checkpoints = list((SUBMISSION / "best_models").glob("*_seed*/best.pt"))
    if len(checkpoints) != 25:
        raise ValueError(f"Expected 25 per-seed best checkpoints, found {len(checkpoints)}")
    snapshots = list(
        (SUBMISSION / "best_models").glob("*_seed*/config.snapshot.json")
    )
    if len(snapshots) != 25:
        raise ValueError(
            f"Expected 25 exact checkpoint configuration snapshots, found {len(snapshots)}"
        )
    metadata = list((SUBMISSION / "best_models").glob("*_seed*/metadata.json"))
    if len(metadata) != 25:
        raise ValueError(f"Expected 25 model metadata records, found {len(metadata)}")
    expected_configs = {
        "random.yaml",
        "sarsa_lambda.yaml",
        "dqn_no_replay.yaml",
        "dqn_replay.yaml",
        "dqn_replay_shaped.yaml",
        "dqn_replay_fast_decay.yaml",
        "final_test.yaml",
        "locked_manifest.yaml",
    }
    actual_configs = {
        path.name for path in (SUBMISSION / "configs").glob("*.yaml")
    }
    if actual_configs != expected_configs:
        raise ValueError(
            f"Submitted locked configuration set differs: {actual_configs}"
        )
    forbidden = [
        path.relative_to(SUBMISSION).as_posix()
        for path in SUBMISSION.rglob("*")
        if "__pycache__" in path.parts
        or path.suffix in {".pyc", ".tmp", ".log"}
        or path.name in {".DS_Store", ".env"}
    ]
    if forbidden:
        raise ValueError(f"Forbidden cache/temporary files in submission: {forbidden}")


def _verify_checksums() -> None:
    """Verify the path-safe SHA-256 manifest against every submitted file."""

    manifest = SUBMISSION / "checksum_manifest.sha256"
    declared: set[Path] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed checksum line {line_number}: {line!r}")
        expected, relative_text = match.groups()
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe checksum path: {relative_text}")
        path = SUBMISSION / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checksummed file is missing: {relative_text}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {relative_text}")
        declared.add(relative)
    actual_files = {
        path.relative_to(SUBMISSION)
        for path in SUBMISSION.rglob("*")
        if path.is_file()
        and path != manifest
        and path.relative_to(SUBMISSION) not in VOLATILE_UNCHECKSUMMED
    }
    if declared != actual_files:
        raise ValueError(
            "Checksum coverage mismatch: "
            f"missing={sorted(actual_files - declared)}, "
            f"extra={sorted(declared - actual_files)}"
        )


def _load_checkpoint_agent(path: Path) -> tuple[dict[str, Any], Agent]:
    """Load one envelope, reconstruct its agent, and restore all model state."""

    payload = load_agent_checkpoint(path, map_location="cpu")
    if payload["config_fingerprint"] != config_fingerprint(payload["config"]):
        raise ValueError(f"Embedded configuration fingerprint mismatch: {path}")
    snapshot = load_config(path.with_name("config.snapshot.json"))
    if snapshot != payload["config"]:
        raise ValueError(f"Submitted configuration snapshot mismatch: {path}")
    config = payload["config"]
    environment = config["environment"]
    seed = int(config["experiment"]["training_seed"])
    env = make_env(seed=seed, max_episode_steps=int(environment["max_episode_steps"]))
    try:
        agent = build_agent(config, env)
    finally:
        env.close()
    agent.load_state_dict(payload["agent_state"])
    return payload, agent


def _verify_models() -> None:
    """Load and reconstruct all 25 validation-selected submitted policies."""

    checkpoints = sorted((SUBMISSION / "best_models").glob("*_seed*/best.pt"))
    seen: set[tuple[str, int]] = set()
    for path in checkpoints:
        payload, _agent = _load_checkpoint_agent(path)
        config = payload["config"]
        metadata_path = path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Model metadata checksum mismatch: {path}")
        if metadata["configuration_hash"] != config_fingerprint(config):
            raise ValueError(f"Model metadata configuration hash mismatch: {path}")
        if metadata["training_seed"] != int(config["experiment"]["training_seed"]):
            raise ValueError(f"Model metadata training seed mismatch: {path}")
        final_metric = metadata.get("final_evaluation_metric", {})
        if final_metric.get("objective") != "original MountainCar-v0 reward and goal":
            raise ValueError(f"Model metadata final objective is invalid: {path}")
        seen.add(
            (
                str(config["experiment"]["condition"]),
                int(config["experiment"]["training_seed"]),
            )
        )
    expected = {
        (condition, seed)
        for condition in CONDITION_IDS - {"random"}
        for seed in range(5)
    }
    if seen != expected:
        raise ValueError(f"Submitted checkpoint condition/seed coverage mismatch: {seen}")


def _verify_locked_configs() -> None:
    """Verify every submitted resolved condition config against the lock manifest."""

    import yaml

    manifest = yaml.safe_load(
        (SUBMISSION / "configs/locked_manifest.yaml").read_text(encoding="utf-8")
    )
    for condition, record in manifest["conditions"].items():
        config = load_config(SUBMISSION / "configs" / f"{condition}.yaml")
        actual = config_fingerprint(config)
        if actual != record["resolved_config_hash"]:
            raise ValueError(f"Submitted locked hash mismatch for {condition}")


def _episode_rows(path: Path) -> list[dict[str, str]]:
    """Read evaluation rows while excluding nondeterministic duration metadata."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("wall_clock_seconds", None)
    return rows


def _verify_deterministic_evaluation() -> None:
    """Repeat a two-episode greedy rollout and require exact saved trajectories."""

    config = load_config(
        SUBMISSION / "best_models/sarsa_lambda_seed0/config.snapshot.json"
    )
    checkpoint = SUBMISSION / "best_models/sarsa_lambda_seed0/best.pt"
    with tempfile.TemporaryDirectory(prefix="mountaincar-submission-") as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        for output in (first, second):
            run_evaluation(
                config,
                output_dir=output,
                split="validation",
                episodes=2,
                overwrite=False,
                checkpoint=checkpoint,
                capture_trajectories_override=True,
            )
        if _episode_rows(first / "episodes.csv") != _episode_rows(second / "episodes.csv"):
            raise ValueError("Deterministic evaluation episode metrics differ")
        if (first / "trajectories.jsonl").read_bytes() != (
            second / "trajectories.jsonl"
        ).read_bytes():
            raise ValueError("Deterministic evaluation trajectories differ")


def _verify_result_tables() -> None:
    """Check table provenance, schema, condition coverage, and summary agreement."""

    submitted_csv = SUBMISSION / "tables/final_results.csv"
    submitted_markdown = SUBMISSION / "tables/final_results.md"
    if submitted_csv.read_bytes() != (ROOT / "results/tables/final_results.csv").read_bytes():
        raise ValueError("Submitted CSV is not the generated final table")
    if submitted_markdown.read_bytes() != (
        ROOT / "results/tables/final_results.md"
    ).read_bytes():
        raise ValueError("Submitted Markdown table is not the generated final table")
    with submitted_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row["condition_id"] for row in rows} != CONDITION_IDS:
        raise ValueError("Final table condition coverage is invalid")
    if any(row["training_seeds"] != "5" for row in rows):
        raise ValueError("Final table must report five training seeds per condition")
    if any(row["evaluation_episodes_per_seed"] != "20" for row in rows):
        raise ValueError("Final table evaluation counts are inconsistent")
    summary = (SUBMISSION / "final_results.md").read_text(encoding="utf-8")
    sarsa = next(row for row in rows if row["condition_id"] == "sarsa_lambda")
    for field in (
        "completion_rate_within_200_steps_95ci",
        "assignment_success_rate_within_100_steps_95ci",
        "median_steps_among_completed_episodes",
    ):
        if sarsa[field] not in summary:
            raise ValueError(f"Submission summary does not match generated field {field}")


def _verify_report_and_figures() -> None:
    """Check report links and basic PNG/PDF integrity and dimensions."""

    from PIL import Image

    for document in (SUBMISSION / "README.md", SUBMISSION / "report.md"):
        text = document.read_text(encoding="utf-8")
        references = re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", text)
        for reference in references:
            if re.match(r"(?:https?://|mailto:|#)", reference):
                continue
            target = (document.parent / reference).resolve()
            if not target.is_file():
                raise FileNotFoundError(
                    f"Broken local path in {document.name}: {reference}"
                )
    for path in sorted((SUBMISSION / "figures").glob("*.png")):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        if width < 1000 or height < 700:
            raise ValueError(f"Figure resolution is too small: {path.name}")
    for path in sorted((SUBMISSION / "figures").glob("*.pdf")):
        data = path.read_bytes()
        if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
            raise ValueError(f"Invalid figure PDF: {path.name}")
        if path.stat().st_size < 10_000:
            raise ValueError(f"Figure PDF is unexpectedly small: {path.name}")
    report_pdf = SUBMISSION / "MountainCar_RL_Recruitment_Report.pdf"
    data = report_pdf.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("Final report PDF is invalid")
    if report_pdf.stat().st_size < 100_000:
        raise ValueError("Final report PDF is unexpectedly small")


def _verify_source_manifest() -> None:
    """Ensure every important source-manifest entry exists in the repository."""

    paths = (SUBMISSION / "source_manifest.txt").read_text(encoding="utf-8").splitlines()
    if not paths:
        raise ValueError("Submission source manifest is empty")
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Source manifest contains missing files: {missing}")


def _verify_text_safety() -> None:
    """Reject placeholders, machine-specific paths, and common secret signatures."""

    patterns = {
        "placeholder": re.compile(r"\b(?:TODO|TBD|lorem ipsum|placeholder result)\b", re.I),
        "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "credential assignment": re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s<]+",
            re.I,
        ),
        "machine-specific path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    }
    for path in SUBMISSION.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".md",
            ".txt",
            ".csv",
            ".yaml",
            ".yml",
            ".json",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                raise ValueError(f"Detected {label} in {path.relative_to(SUBMISSION)}")
    audit = (ROOT / "AUDIT_REPORT.md").read_text(encoding="utf-8")
    if re.search(r"unresolved\s+(?:critical|high)", audit, re.I):
        raise ValueError("AUDIT_REPORT.md contains an unresolved critical/high issue")


def _verify_external_pdf_tools() -> None:
    """Use pdfinfo when available for an additional final-report open check."""

    command = ["pdfinfo", str(SUBMISSION / "MountainCar_RL_Recruitment_Report.pdf")]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return
    if "Pages:" not in completed.stdout:
        raise ValueError("pdfinfo did not report a page count")


def main() -> int:
    """Run all fast submission integrity checks."""

    _verify_required_files()
    _verify_checksums()
    _verify_locked_configs()
    _verify_models()
    _verify_deterministic_evaluation()
    _verify_result_tables()
    _verify_report_and_figures()
    _verify_source_manifest()
    _verify_text_safety()
    _verify_external_pdf_tools()
    print(
        "PASS: required files, checksums, locked hashes, 25 best models and "
        "metadata records, deterministic evaluation, result tables, report "
        "paths/PDFs, source manifest, text safety, and audit status verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
