"""Checks that enforce the repository's layering and config rules."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "evenet_align" / "core"
SOURCES = [p for d in ("evenet_align", "scripts") for p in (ROOT / d).rglob("*.py")]


def test_core_does_no_io_or_printing():
    pattern = re.compile(r"(?<![\w.])(open|print|input)\(|\.write_text\(|\.read_text\(|torch\.save\(|torch\.load\(")
    offenders = [f"{p.name}:{i}" for p in CORE.rglob("*.py")
                 for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert not offenders, offenders


def test_no_mode_branches():
    pattern = re.compile(r"\bif\s+.*\b(smoke|debug|nersc)\b", re.IGNORECASE)
    offenders = [f"{p.relative_to(ROOT)}:{i}" for p in SOURCES
                 for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert not offenders, offenders


@pytest.mark.skipif(shutil.which("lint-imports") is None, reason="import-linter not installed")
def test_import_contracts():
    result = subprocess.run(["lint-imports"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
