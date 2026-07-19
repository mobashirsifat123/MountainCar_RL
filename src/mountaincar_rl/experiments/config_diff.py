"""Generate auditable, machine-readable resolved configuration comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mountaincar_rl.utils.config import load_config
from mountaincar_rl.utils.logging import write_json


_MISSING = object()


def _fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configuration_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return sorted leaf-level differences between two resolved configs."""

    differences: list[dict[str, Any]] = []

    def visit(path: str, left: Any, right: Any) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                visit(
                    f"{path}.{key}" if path else str(key),
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                )
            return
        if left is not _MISSING and right is not _MISSING and left == right:
            return
        differences.append(
            {
                "key": path,
                "reference": None if left is _MISSING else left,
                "candidate": None if right is _MISSING else right,
                "reference_missing": left is _MISSING,
                "candidate_missing": right is _MISSING,
            }
        )

    visit("", reference, candidate)
    return differences


def build_comparison_report(manifest_path: Path) -> dict[str, Any]:
    """Resolve every manifest group and compare each candidate to its reference."""

    manifest = load_config(manifest_path)
    groups = manifest.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("Comparison manifest must contain a nonempty 'groups' mapping")
    base_dir = manifest_path.resolve().parent
    report_groups: dict[str, Any] = {}
    for group_name, group_value in groups.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("Comparison group names must be nonempty strings")
        if not isinstance(group_value, Mapping):
            raise TypeError(f"Comparison group {group_name!r} must be a mapping")
        reference_value = group_value.get("reference")
        candidates = group_value.get("candidates")
        if not isinstance(reference_value, str) or not reference_value:
            raise TypeError(f"Comparison group {group_name!r} needs a reference path")
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError(
                f"Comparison group {group_name!r} needs nonempty candidates"
            )
        reference_path = (base_dir / reference_value).resolve()
        reference = load_config(reference_path)
        candidate_reports: dict[str, Any] = {}
        for label, candidate_value in candidates.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Candidate labels must be nonempty strings")
            if not isinstance(candidate_value, str) or not candidate_value:
                raise TypeError(f"Candidate {label!r} must name a config path")
            candidate_path = (base_dir / candidate_value).resolve()
            candidate = load_config(candidate_path)
            differences = configuration_differences(reference, candidate)
            candidate_reports[label] = {
                "path": candidate_value,
                "resolved_sha256": _fingerprint(candidate),
                "difference_count": len(differences),
                "differences": differences,
            }
        report_groups[group_name] = {
            "reference": {
                "path": reference_value,
                "resolved_sha256": _fingerprint(reference),
            },
            "candidates": candidate_reports,
        }
    return {
        "schema_version": 1,
        "manifest": manifest_path.as_posix(),
        "comparison_basis": "fully resolved inherited configurations",
        "groups": report_groups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_comparison_report(args.manifest)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_comparison_report", "configuration_differences"]
