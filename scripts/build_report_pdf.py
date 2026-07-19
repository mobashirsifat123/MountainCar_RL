"""Build and mechanically validate the professor-facing PDF report."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/report.md"
CSS = ROOT / "report/report.css"
TEMP_DIR = ROOT / "tmp/pdfs"
OUTPUT_DIR = ROOT / "output/pdf"
OUTPUT = OUTPUT_DIR / "MountainCar_RL_Recruitment_Report.pdf"
LOG = OUTPUT_DIR / "report_build_log.txt"


def _command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([item]) for item in command)


def main() -> int:
    """Convert Markdown through Pandoc HTML and WeasyPrint, then validate it."""

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = TEMP_DIR / "report.html"
    lines = [
        "MountainCar report PDF build",
        f"Python: {sys.version.split()[0]}",
        f"Source: {REPORT.relative_to(ROOT)}",
        f"CSS: {CSS.relative_to(ROOT)}",
    ]
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        lines.append("FAIL: pandoc executable not found")
        LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError("pandoc is required to build the report PDF")
    command = [
        pandoc,
        REPORT.relative_to(ROOT).as_posix(),
        "--from=markdown+tex_math_single_backslash+tex_math_dollars",
        "--to=html5",
        "--standalone",
        "--mathml",
        "--embed-resources",
        "--resource-path=.:report:results",
        f"--css={CSS.relative_to(ROOT).as_posix()}",
        "--metadata=pagetitle:MountainCar RL Recruitment Report",
        f"--output={html_path.relative_to(ROOT).as_posix()}",
    ]
    lines.append(f"Pandoc command: {_command_text(command)}")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            lines.append(f"Pandoc stdout:\n{completed.stdout.rstrip()}")
        if completed.stderr:
            lines.append(f"Pandoc stderr:\n{completed.stderr.rstrip()}")
        lines.append(
            "WeasyPrint command: uv run --python 3.12 --with weasyprint "
            "--with pypdf python scripts/build_report_pdf.py"
        )
        HTML(filename=html_path.as_posix(), base_url=ROOT.as_posix()).write_pdf(
            OUTPUT.as_posix()
        )
        reader = PdfReader(OUTPUT)
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        required_text = (
            "Learning to Build Momentum",
            "No condition reliably met",
            "SARSA",
            "Reproducibility",
        )
        missing = [text for text in required_text if text not in extracted]
        if missing:
            raise ValueError(f"Generated PDF is missing required text: {missing}")
        if OUTPUT.stat().st_size < 100_000:
            raise ValueError("Generated PDF is unexpectedly small")
        lines.extend(
            [
                f"Output: {OUTPUT.relative_to(ROOT)}",
                f"Pages: {len(reader.pages)}",
                f"Bytes: {OUTPUT.stat().st_size}",
                "Text extraction: PASS",
                "Build result: PASS",
            ]
        )
    except Exception as error:
        lines.append(f"Build result: FAIL: {type(error).__name__}: {error}")
        LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    print(LOG.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
