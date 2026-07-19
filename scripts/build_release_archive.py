"""Create and extract-verify the clean professor-facing ZIP archive."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from mountaincar_rl.utils.checkpoints import config_fingerprint, load_agent_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
ARCHIVE = ROOT / "MountainCar_RL_Recruitment_Submission.zip"
REPORT = SUBMISSION / "archive_verification.md"
SIZE_TOKEN = "0000000000"
REQUIRED = {
    "README.md",
    "MountainCar_RL_Recruitment_Report.pdf",
    "report.md",
    "final_results.md",
    "requirements.txt",
    "reproduction_commands.txt",
    "verification_output.txt",
    "figure_manifest.md",
    "environment.txt",
    "source_manifest.txt",
    "checksum_manifest.sha256",
}


def _report(size_text: str) -> str:
    return f"""# Archive verification

- Archive: `MountainCar_RL_Recruitment_Submission.zip`
- Archive size: {size_text} bytes
- Archive opens and CRC test passes: PASS
- Required top-level files present after extraction: PASS
- Submitted checksum manifest verifies after extraction: PASS
- Final report PDF header/trailer and size: PASS
- All 25 checkpoint envelopes load and configuration fingerprints match: PASS
- Reproduction commands present: PASS
- Forbidden cache/secret filenames absent: PASS

The archive contains the contents of `submission/` at its root. The archive
verification record and verification output are intentionally excluded from the
scientific checksum manifest because they are generated after integrity checks;
all reports, figures, tables, configs, model artifacts, and model metadata are
checksummed.
"""


def _write_zip() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as archive:
        for path in sorted(SUBMISSION.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(SUBMISSION).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path == REPORT
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, relative, compress_type=compression, compresslevel=6)


def _verify_checksums(extracted: Path) -> None:
    manifest = extracted / "checksum_manifest.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed extracted checksum line: {line!r}")
        expected, relative = match.groups()
        path = extracted / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Extracted checksum mismatch: {relative}")


def _verify_extracted(extracted: Path) -> None:
    missing = sorted(name for name in REQUIRED if not (extracted / name).is_file())
    if missing:
        raise FileNotFoundError(f"Extracted archive lacks files: {missing}")
    _verify_checksums(extracted)
    pdf = extracted / "MountainCar_RL_Recruitment_Report.pdf"
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("Extracted report PDF failed structural checks")
    if pdf.stat().st_size < 100_000:
        raise ValueError("Extracted report PDF is unexpectedly small")
    checkpoints = sorted((extracted / "best_models").glob("*_seed*/best.pt"))
    if len(checkpoints) != 25:
        raise ValueError(f"Expected 25 checkpoints, found {len(checkpoints)}")
    for checkpoint in checkpoints:
        payload = load_agent_checkpoint(checkpoint, map_location="cpu")
        if payload["config_fingerprint"] != config_fingerprint(payload["config"]):
            raise ValueError(f"Checkpoint fingerprint mismatch: {checkpoint}")
    if "make verify-submission" not in (
        extracted / "reproduction_commands.txt"
    ).read_text(encoding="utf-8"):
        raise ValueError("Reproduction commands omit submission verification")
    forbidden = [
        path.relative_to(extracted).as_posix()
        for path in extracted.rglob("*")
        if "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pem"}
        or path.name in {".env", ".DS_Store", "id_rsa", "id_ed25519"}
    ]
    if forbidden:
        raise ValueError(f"Forbidden archive files: {forbidden}")


def main() -> int:
    """Build twice with a fixed-width size record, then extract and verify."""

    REPORT.write_text(_report(SIZE_TOKEN), encoding="utf-8")
    _write_zip()
    initial_size = ARCHIVE.stat().st_size
    size_text = f"{initial_size:010d}"
    if len(size_text) != len(SIZE_TOKEN):
        raise ValueError("Archive size exceeds fixed-width verification field")
    REPORT.write_text(_report(size_text), encoding="utf-8")
    _write_zip()
    if ARCHIVE.stat().st_size != initial_size:
        raise ValueError("Final archive size changed after fixed-width report update")
    with zipfile.ZipFile(ARCHIVE) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC failure: {corrupt}")
        with tempfile.TemporaryDirectory(prefix="mountaincar-archive-") as temporary:
            extracted = Path(temporary)
            archive.extractall(extracted)
            _verify_extracted(extracted)
    print(f"Archive: {ARCHIVE.name}")
    print(f"Bytes: {ARCHIVE.stat().st_size}")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
