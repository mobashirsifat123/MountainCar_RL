"""Atomic checkpoint persistence for model and optimizer state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


def _checkpoint_target(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_save_bytes(path: str | Path, data: bytes) -> Path:
    """Atomically replace ``path`` with ``data`` from the same filesystem."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    target = _checkpoint_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def save_torch_checkpoint(path: str | Path, payload: Any) -> Path:
    """Atomically save a PyTorch-serializable checkpoint payload."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Saving a torch checkpoint requires PyTorch") from error

    target = _checkpoint_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    """Alias for :func:`save_torch_checkpoint`."""

    return save_torch_checkpoint(path, payload)


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
    weights_only: bool = False,
) -> Any:
    """Load a checkpoint with an explicit device and deserialization mode."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Loading a torch checkpoint requires PyTorch") from error

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
    )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    try:
        canonical = json.dumps(
            dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("config must contain only finite JSON-compatible data") from error
    return hashlib.sha256(canonical).hexdigest()


def save_agent_checkpoint(
    path: str | Path,
    *,
    agent_type: str,
    agent_state: Mapping[str, Any],
    config: Mapping[str, Any],
    training_state: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> Path:
    """Save a versioned, self-describing agent/training checkpoint atomically.

    ``agent_state`` owns algorithm-specific network/optimizer/RNG/replay or
    weight/trace data. The shared envelope pins the full configuration identity,
    preprocessing metadata, completed progress, and validation selection record.
    """

    if not isinstance(agent_type, str) or not agent_type:
        raise TypeError("agent_type must be a nonempty string")
    for name, value in (
        ("agent_state", agent_state),
        ("config", config),
        ("training_state", training_state),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "agent_type": agent_type,
        "config_fingerprint": config_fingerprint(config),
        "config": dict(config),
        "preprocessing": None if preprocessing is None else dict(preprocessing),
        "agent_state": dict(agent_state),
        "training_state": dict(training_state),
        "selection": None if selection is None else dict(selection),
    }
    return save_torch_checkpoint(path, payload)


def load_agent_checkpoint(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
    expected_agent_type: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate the shared checkpoint envelope.

    Configuration validation prevents silently evaluating weights with a
    different architecture, preprocessing contract, or reward treatment.
    """

    payload = load_torch_checkpoint(
        path, map_location=map_location, weights_only=False
    )
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint root must be a mapping")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema version: "
            f"{payload.get('checkpoint_schema_version')!r}"
        )
    required = {
        "agent_type",
        "config_fingerprint",
        "config",
        "agent_state",
        "training_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    if expected_agent_type is not None and payload["agent_type"] != expected_agent_type:
        raise ValueError(
            f"Checkpoint agent_type {payload['agent_type']!r} does not match "
            f"expected {expected_agent_type!r}"
        )
    if expected_config is not None:
        expected_fingerprint = config_fingerprint(expected_config)
        if payload["config_fingerprint"] != expected_fingerprint:
            raise ValueError(
                "Checkpoint configuration fingerprint does not match the "
                "requested configuration"
            )
    return payload
