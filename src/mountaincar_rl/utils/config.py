"""Strict loading and lightweight validation of experiment configurations."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:  # Python 3.11+; ``tomli`` is the compatibility dependency on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and resolve a YAML, TOML, or JSON mapping.

    A configuration may declare one relative parent with ``extends: path``.
    Nested mappings are merged recursively while scalar values and sequences
    replace their parent values. ``extends`` is removed from the returned
    resolved mapping, so saved snapshots and checkpoint fingerprints describe
    the actual run rather than depending on later parent-file changes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported.
        TypeError: If the parsed document is not a top-level mapping.
        RuntimeError: If YAML is requested but PyYAML is unavailable.
    """

    return _load_config(Path(path), stack=())


def _load_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    """Resolve one config and its parent while rejecting inheritance cycles."""

    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Configuration inheritance cycle detected: {cycle}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".toml":
        try:
            parsed: Any = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(
                f"Failed to parse configuration {config_path}: {error}"
            ) from error
    elif suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Failed to parse configuration {config_path}: {error}"
            ) from error
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Loading YAML configuration requires the 'PyYAML' package"
            ) from error
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Failed to parse configuration {config_path}: {error}"
            ) from error
    else:
        raise ValueError(
            f"Unsupported configuration format '{suffix or '<none>'}'; "
            "expected .yaml, .yml, .toml, or .json"
        )

    if not isinstance(parsed, Mapping):
        raise TypeError(
            f"Configuration root must be a mapping, got {type(parsed).__name__}"
        )
    child = dict(parsed)
    parent_value = child.pop("extends", None)
    if parent_value is None:
        return deepcopy(child)
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise TypeError("Configuration 'extends' must be a nonempty path string")
    parent_path = (config_path.parent / parent_value).resolve()
    parent = _load_config(parent_path, stack=(*stack, config_path))
    return _deep_merge(parent, child)


def _deep_merge(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge mappings; child sequences and scalars replace parents."""

    merged = deepcopy(dict(parent))
    for key, child_value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, Mapping) and isinstance(child_value, Mapping):
            merged[key] = _deep_merge(parent_value, child_value)
        else:
            merged[key] = deepcopy(child_value)
    return merged


def require_keys(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
    *,
    context: str = "config",
) -> None:
    """Raise ``KeyError`` listing all required keys absent from ``mapping``."""

    if not isinstance(mapping, Mapping):
        raise TypeError(f"{context} must be a mapping")
    required = tuple(keys)
    if any(not isinstance(key, str) for key in required):
        raise TypeError("required keys must be strings")
    missing = sorted(key for key in required if key not in mapping)
    if missing:
        formatted = ", ".join(repr(key) for key in missing)
        raise KeyError(f"Missing required {context} key(s): {formatted}")


def require_type(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
    *,
    context: str = "config",
) -> Any:
    """Return a required value after checking its runtime type."""

    require_keys(mapping, (key,), context=context)
    value = mapping[key]
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_name = " or ".join(candidate.__name__ for candidate in expected_type)
        else:
            type_name = expected_type.__name__
        raise TypeError(
            f"{context}.{key} must be {type_name}, got {type(value).__name__}"
        )
    return value
