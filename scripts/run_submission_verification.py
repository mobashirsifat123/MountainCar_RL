"""Run final submission checks and atomically save their complete output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission/verification_output.txt"
COMMANDS = (
    ["uv", "run", "--python", "3.12", "python", "scripts/verify_submission.py"],
    ["uv", "run", "--python", "3.12", "pytest", "-q"],
)


def main() -> int:
    """Run each check in order, record stdout/stderr, and preserve failures."""

    lines: list[str] = []
    status = 0
    for command in COMMANDS:
        rendered = " ".join(command)
        lines.append(f"$ {rendered}")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        lines.append(completed.stdout.rstrip())
        if completed.stderr:
            lines.append(completed.stderr.rstrip())
        lines.append(f"exit_code={completed.returncode}")
        if completed.returncode != 0:
            status = completed.returncode
            break
    text = "\n".join(line for line in lines if line) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(OUTPUT)
    print(text, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
