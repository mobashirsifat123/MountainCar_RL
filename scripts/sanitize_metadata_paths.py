"""Replace machine-specific repository prefixes in generated metadata files."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLACEMENT = "$REPOSITORY_ROOT"


def sanitize(*, check: bool) -> tuple[int, list[Path]]:
    """Sanitize metadata text without changing experimental metric content."""

    changed: list[Path] = []
    prefix = ROOT.as_posix()
    for path in sorted((ROOT / "results").rglob("metadata.json")):
        text = path.read_text(encoding="utf-8")
        updated = text.replace(prefix, REPLACEMENT)
        if updated == text:
            continue
        changed.append(path)
        if not check:
            path.write_text(updated, encoding="utf-8")
    return len(changed), changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count, paths = sanitize(check=args.check)
    if args.check and count:
        print(f"FAIL: {count} metadata files retain the local repository path")
        for path in paths:
            print(path.relative_to(ROOT))
        return 1
    if args.check:
        print("PASS: generated metadata contains no local repository path")
    else:
        print(f"Sanitized {count} generated metadata files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
