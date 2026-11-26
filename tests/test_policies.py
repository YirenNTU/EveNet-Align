"""Policy backends: the MLP interface, the supervised-first rule for EveNet, and an optional EveNet run."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from evenet_align.core.models import build_mlp_policy, mlp_statistics
from evenet_align.core.pseudo_data import pseudo_tau
from evenet_align.core.types import MLPConfig
from evenet_align.experiments import policies
from evenet_align.experiments.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mlp_policy_round_trips_targets():
    events = pseudo_tau(200, 1)
    policy = build_mlp_policy(8, (2, 2), MLPConfig(hidden=16, depth=1), mlp_statistics(events), seed=0)
    x = policy.normalize(events.target)
    assert x.shape == (200, 4)
    assert torch.allclose(policy.denormalize(x), events.target, atol=1e-6)
    v = policy.velocity(x, events.observed, torch.rand(200))
    assert v.shape == (200, 4)
    assert policy.candidate_features(events.target, events.observed).shape == (200, 0)


def _evenet_config(tmp_path, checkpoint):
    for name in ("config.yaml", "normalization.pt"):
        (tmp_path / name).write_bytes(b"placeholder")
    return load_config(ROOT / "configs" / "ztautau.yaml", [
        f"policy.evenet.checkpoint={checkpoint}", f"policy.evenet.config={tmp_path / 'config.yaml'}",
        f"policy.evenet.normalization={tmp_path / 'normalization.pt'}"])


def test_released_pretrained_weights_are_refused(tmp_path, monkeypatch):
    checkpoint = tmp_path / "released.ckpt"
    checkpoint.write_bytes(b"pretend these are the released weights")
    monkeypatch.setitem(policies.RELEASED_PRETRAINED, policies.sha256(checkpoint), "SSL.20M.last.ckpt")
    with pytest.raises(ValueError, match="supervised"):
        policies.BACKENDS["evenet"].inputs(_evenet_config(tmp_path, checkpoint), 0)


def test_finetuned_checkpoint_digest_enters_the_run_hash(tmp_path):
    checkpoint = tmp_path / "finetuned.ckpt"
    checkpoint.write_bytes(b"fine-tuned weights")
    inputs = policies.BACKENDS["evenet"].inputs(_evenet_config(tmp_path, checkpoint), 0)
    assert inputs["evenet_checkpoint_sha256"] == policies.sha256(checkpoint)


@pytest.mark.evenet
@pytest.mark.skipif(not os.environ.get("EVENET_SHARE"), reason="set EVENET_SHARE to an EveNet share/ directory")
def test_evenet_policy_runs_dgpo_end_to_end(tmp_path):
    """Tiny random EveNet 'fine-tune' + EveNet-format pseudo ttbar events through the full DGPO run."""
    pytest.importorskip("evenet.network.evenet_model")
    fixture = subprocess.run([sys.executable, str(ROOT / "tests" / "evenet_fixture.py"), os.environ["EVENET_SHARE"],
                              str(tmp_path / "fixture")], capture_output=True, text=True)
    assert fixture.returncode == 0, fixture.stderr
    f = tmp_path / "fixture"
    sets = ["policy.name=evenet", f"policy.evenet.checkpoint={f / 'finetuned.ckpt'}",
            f"policy.evenet.config={f / 'share' / 'tiny-finetune.yaml'}", f"policy.evenet.normalization={f / 'normalization.pt'}",
            "dataset.source=npz", f"dataset.path={f / 'data'}", "dataset.observed=[x,x_mask,conditions,conditions_mask]",
            "dataset.target=x_invisible", "oracle.function=evenet_align.examples.ttbar:top_momentum_oracle",
            "oracle.params={}", "classifier.features.function=evenet_align.examples.ttbar:top_features",
            "dgpo.batch_size=128", "classifier.fit_events=1024", "classifier.batch_size=256", "dgpo.steps=10"]
    runs = []
    for results in ("a", "b"):
        command = [sys.executable, str(ROOT / "scripts" / "run.py"), str(ROOT / "configs" / "smoke.yaml"),
                   "--results", str(tmp_path / results)] + [x for s in sets for x in ("--set", s)]
        out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-3000:]
        runs.append(Path(out.stdout.strip().splitlines()[-1]))
    for step in ("step_0000000.npz", "step_0000010.npz"):
        a, b = (np.load(r / "validation" / step)["prediction"] for r in runs)
        np.testing.assert_array_equal(a, b)
    inputs = json.loads((runs[0] / "provenance.json").read_text())["inputs"]
    assert set(inputs) == {"evenet_checkpoint_sha256", "evenet_config_sha256", "evenet_normalization_sha256",
                           "evenet_source"}


CUSTOM = ["policy.name=custom", "policy.custom.function=evenet_align.examples.custom_policy:mlp_policy",
          "policy.custom.params={hidden: 16, depth: 1, time_frequencies: 4}"]


def test_custom_policy_loads_and_is_checked():
    events = pseudo_tau(200, 1)
    cfg = load_config(ROOT / "configs" / "smoke.yaml", CUSTOM)
    policy = policies.BACKENDS["custom"].load(cfg, 0, ROOT / "unused", "cpu", events)
    assert isinstance(policy, torch.nn.Module) and tuple(policy.target_shape) == tuple(events.target.shape[1:])
    bad = load_config(ROOT / "configs" / "smoke.yaml", CUSTOM[:1] + ["policy.custom.function=builtins:id"])
    with pytest.raises(TypeError, match="nn.Module"):
        policies.BACKENDS["custom"].load(bad, 0, ROOT / "unused", "cpu", events)
    with pytest.raises(ValueError, match="policy.custom.function"):
        load_config(ROOT / "configs" / "smoke.yaml", ["policy.name=custom"])


def test_custom_policy_files_enter_the_run_hash(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"a")
    cfg = load_config(ROOT / "configs" / "smoke.yaml", CUSTOM + [f"policy.custom.files=[{weights}]"])
    first = policies.BACKENDS["custom"].inputs(cfg, 0)
    weights.write_bytes(b"b")
    assert policies.BACKENDS["custom"].inputs(cfg, 0) != first


@pytest.mark.slow
def test_custom_policy_runs_dgpo_end_to_end(tmp_path):
    command = [sys.executable, str(ROOT / "scripts" / "run.py"), str(ROOT / "configs" / "smoke.yaml"),
               "--results", str(tmp_path), "--set", "dgpo.steps=4", "--set", "dgpo.refit_every=2"]
    for o in CUSTOM:
        command += ["--set", o]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    run_dir = Path(out.stdout.strip().splitlines()[-1])
    assert (run_dir / "COMPLETE").is_file() and (run_dir / "policy_final.pt").is_file()
