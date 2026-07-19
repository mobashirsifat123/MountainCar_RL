"""Capture provenance needed to interpret and reproduce an experiment run."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


_PACKAGES = {
    "gymnasium": "gymnasium",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
    "pyyaml": "PyYAML",
    "torch": "torch",
}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, distribution in _PACKAGES.items():
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _run_git(arguments: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    return completed.stdout.strip()


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        root = Path(_run_git(("rev-parse", "--show-toplevel"), cwd))
        commit = _run_git(("rev-parse", "HEAD"), root)
        branch = _run_git(("rev-parse", "--abbrev-ref", "HEAD"), root)
        porcelain = _run_git(("status", "--porcelain"), root)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
            "root": None,
        }
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(porcelain),
        "root": str(root),
    }


def _serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_run_metadata(
    config: Mapping[str, Any], device: str = "cpu"
) -> dict[str, Any]:
    """Collect immutable run context without assuming execution inside Git."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a nonempty string")

    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "device": device,
        "packages": _package_versions(),
        "git": _git_metadata(Path.cwd()),
        "config": _serializable(dict(config)),
    }
