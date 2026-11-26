"""End-to-end runs of configs/smoke.yaml: completion, provenance, determinism, kill/rerun."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
pytestmark = pytest.mark.slow


def run(results: Path, *overrides: str) -> Path:
    command = [sys.executable, str(ROOT / "scripts" / "run.py"), str(SMOKE), "--results", str(results)]
    for o in overrides:
        command += ["--set", o]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return Path(out.stdout.strip().splitlines()[-1])


def outputs(run_dir: Path) -> dict[str, np.ndarray]:
    """Every result array of a DGPO run (provenance excluded)."""
    arrays = {p.name: np.load(p)["prediction"] for p in sorted((run_dir / "validation").glob("step_*.npz"))}
    arrays.update({f"policy/{k}": v.numpy() for k, v in
                   torch.load(run_dir / "policy_final.pt", weights_only=True).items()})
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    arrays["metrics"] = np.array([v for m in metrics for k, v in sorted(m.items()) if k != "stage"], dtype=float)
    return arrays


def assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    different = {k: float(np.abs(a[k] - b[k]).max()) for k in a if not np.array_equal(a[k], b[k])}
    assert not different, different


@pytest.fixture(scope="module")
def clean(tmp_path_factory) -> Path:
    return run(tmp_path_factory.mktemp("clean"))


def test_run_completes_with_expected_layout(clean):
    assert (clean / "COMPLETE").is_file()
    names = sorted(p.name for p in (clean / "validation").iterdir())
    assert names == ["step_0000000.npz", "step_0000010.npz", "step_0000020.npz", "truth.npz"]
    assert sorted(p.name for p in (clean / "classifier").iterdir()) == ["round_00.pt", "round_01.pt"]
    stages = [json.loads(line)["stage"] for line in (clean / "metrics.jsonl").read_text().splitlines()]
    assert stages[:3] == ["validation", "classifier", "dgpo"]  # validate, then refit, then update
    assert stages.count("dgpo") == 20
    assert not (clean / "checkpoint").exists()


def test_provenance_fields(clean):
    p = json.loads((clean / "provenance.json").read_text())
    for field in ("run_id", "seed", "config", "git_commit", "git_dirty", "python", "packages", "nersc_modules"):
        assert field in p, field
    assert p["config"]["dgpo"]["candidates"] == 4 and p["seed"] == 0


def test_same_config_and_seed_reproduce(clean, tmp_path):
    assert_same(outputs(clean), outputs(run(tmp_path)))


def test_finished_run_is_skipped(clean):
    before = (clean / "COMPLETE").stat().st_mtime_ns
    run(clean.parents[1])
    assert (clean / "COMPLETE").stat().st_mtime_ns == before


def _kill_when(ready, results: Path, *overrides: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "run.py"), str(SMOKE), "--results", str(results)]
    for o in overrides:
        command += ["--set", o]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while not ready() and process.poll() is None and time.time() < deadline:
        time.sleep(0.02)
    assert process.poll() is None, "run finished before it could be killed"
    os.kill(process.pid, signal.SIGKILL)
    process.wait()


def _fit_in_progress(path: Path) -> bool:
    try:
        saved = torch.load(path, weights_only=True)
    except (FileNotFoundError, RuntimeError, EOFError):
        return False
    return saved["step"] == 10 and not saved["state"]["finished"]


def test_killed_after_policy_checkpoint_resumes_to_same_state(clean, tmp_path):
    partial = tmp_path / "smoke" / ".partial" / clean.name
    _kill_when(lambda: (partial / "validation" / "step_0000010.npz").exists(), tmp_path)
    assert not (tmp_path / "smoke" / clean.name).exists() and (partial / "checkpoint" / "state.pt").exists()
    rerun = run(tmp_path)
    assert json.loads((rerun / "provenance.json").read_text())["resumed_at_steps"]
    assert_same(outputs(clean), outputs(rerun))


def test_killed_during_classifier_fit_resumes_to_same_state(tmp_path):
    longer_fit = "classifier.min_updates=300"  # long enough to interrupt reliably
    reference = run(tmp_path / "clean", longer_fit)
    partial = tmp_path / "killed" / "smoke" / ".partial" / reference.name
    _kill_when(lambda: _fit_in_progress(partial / "checkpoint" / "fit.pt"), tmp_path / "killed", longer_fit)
    assert _fit_in_progress(partial / "checkpoint" / "fit.pt")
    rerun = run(tmp_path / "killed", longer_fit)
    assert json.loads((rerun / "provenance.json").read_text())["resumed_at_steps"] == [10]
    assert_same(outputs(reference), outputs(rerun))


def test_npz_source_matches_pseudo_source(clean, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_pseudo_npz.py"), str(SMOKE), str(tmp_path / "data")],
                   cwd=ROOT, check=True, capture_output=True)
    from_npz = run(tmp_path / "results", "dataset.source=npz", f"dataset.path={tmp_path / 'data'}")
    a, b = outputs(clean), outputs(from_npz)
    assert_same({k: v for k, v in a.items() if k.startswith("step")}, {k: v for k, v in b.items() if k.startswith("step")})


def test_ttbar_smoke_and_summary(tmp_path):
    run_dir = run(tmp_path / "results", "dataset.pseudo=ttbar",
                  "oracle.function=evenet_align.examples.ttbar:top_momentum_oracle", "oracle.params={}",
                  "classifier.features.function=evenet_align.examples.ttbar:top_features")
    assert (run_dir / "COMPLETE").is_file()
    from evenet_align.analysis.summarize import summarize

    out = summarize(run_dir.parent, tmp_path / "figures")
    rows = (out / "summary.csv").read_text().splitlines()
    assert rows[0].startswith("config,step,metric") and any("vote_distance_mean" in r for r in rows)


def test_never_writes_into_foreign_directories(clean, tmp_path):
    foreign = tmp_path / "paper_results"
    foreign.mkdir()
    (foreign / "existing_result.csv").write_text("keep me\n")
    command = [sys.executable, str(ROOT / "scripts" / "run.py"), str(SMOKE), "--results", str(foreign)]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 and "not written by evenet-align" in out.stderr
    assert sorted(p.name for p in foreign.iterdir()) == ["existing_result.csv"]
    from evenet_align.analysis.summarize import summarize

    with pytest.raises(FileExistsError):
        summarize(clean.parent, foreign)
    assert (foreign / "existing_result.csv").read_text() == "keep me\n"
