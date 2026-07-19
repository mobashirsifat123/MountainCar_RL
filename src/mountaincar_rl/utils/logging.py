"""Structured metric persistence with atomic replacement where applicable."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def _atomic_text_writer(target: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline=newline
        ) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, payload: Any) -> Path:
    """Serialize ``payload`` as strict JSON and atomically replace ``path``."""

    target = _ensure_parent(path)
    with _atomic_text_writer(target) as handle:
        json.dump(
            payload,
            handle,
            default=_json_default,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return target


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one strict-JSON record and durably flush it to disk."""

    if not isinstance(record, Mapping):
        raise TypeError("JSONL record must be a mapping")
    target = _ensure_parent(path)
    line = json.dumps(
        dict(record),
        default=_json_default,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    """Write mappings as a headered CSV file using atomic replacement."""

    columns = tuple(fieldnames)
    if not columns:
        raise ValueError("fieldnames must not be empty")
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError("fieldnames must contain nonempty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("fieldnames must be unique")

    target = _ensure_parent(path)
    with _atomic_text_writer(target, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("each CSV row must be a mapping")
            writer.writerow(dict(row))
    return target
