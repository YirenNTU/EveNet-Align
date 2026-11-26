"""Run directories: one writer each, atomic completion, provenance next to outputs.

A run is built in ``results/<experiment>/.partial/<run_id>/``. When every output
is written, the ``COMPLETE`` marker goes in last and the directory is renamed to
``results/<experiment>/<run_id>/``. A finished run (marker present) is skipped.
An interrupted run that saved a resumable checkpoint continues from it; one
without a checkpoint is deleted and redone from scratch. Either way the final
state equals that of an uninterrupted run.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

MARKER = "COMPLETE"
OWNER = ".evenet-align-results"  # marks a results root this package created
PACKAGES = ("torch", "numpy", "pyyaml", "evenet-align")


def final_dir(results: Path, experiment: str, run_id: str) -> Path:
    return results / experiment / run_id


def is_complete(path: Path) -> bool:
    return (path / MARKER).is_file()


def claim_root(results: Path) -> None:
    """Use ``results`` only if it is new, empty, or was created by this package; never write into other outputs."""
    if results.exists() and any(results.iterdir()) and not (results / OWNER).is_file():
        raise FileExistsError(f"{results} already holds files not written by evenet-align; choose a new --results")
    results.mkdir(parents=True, exist_ok=True)
    (results / OWNER).touch()


def prepare(results: Path, experiment: str, run_id: str, checkpoint: str | None) -> tuple[Path, bool]:
    """Working directory for a run and whether it resumes from ``<partial>/<checkpoint>``.

    Only this run's own partial directory is ever deleted; an existing final
    directory without its marker is reported, not removed.
    """
    claim_root(results)
    final = final_dir(results, experiment, run_id)
    partial = results / experiment / ".partial" / run_id
    if final.exists():
        raise FileExistsError(f"{final} exists without a {MARKER} marker; inspect it and move it away yourself")
    if checkpoint is not None and (partial / checkpoint).is_file():
        return partial, True
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    return partial, False


def publish(partial: Path, results: Path, experiment: str, run_id: str) -> Path:
    (partial / MARKER).write_text("ok\n")
    final = final_dir(results, experiment, run_id)
    os.replace(partial, final)
    return final


def save_torch(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _git(*args: str) -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(resolved_config: dict, seed: int, run_id: str, device: str) -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "run_id": run_id,
        "seed": seed,
        "config": resolved_config,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "packages": versions,
        "nersc_modules": os.environ.get("LOADEDMODULES", "").split(":") if os.environ.get("LOADEDMODULES") else [],
    }
