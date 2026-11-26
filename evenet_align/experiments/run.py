"""Thin shell: config -> events -> supervised policy -> DGPO -> run directory.

The supervised starting policy comes from ``policies.BACKENDS`` (a fine-tuned
EveNet checkpoint, or the reference MLP's own supervised run). Each seed is one
resumable run ``dgpo-<hash>-seed<s>``; the hash covers the config and the
digest of the supervised policy it starts from.

DGPO run directory:
    config.yaml, provenance.json   resolved config and environment (resume steps appended)
    metrics.jsonl                  one JSON object per record (stage, step, values)
    setup.pt                       oracle constants and pinned classifier feature standardization
    validation/truth.npz           validation MC targets and event ids
    validation/step_XXXXXXX.npz    one raw DDIM draw per validation event
    classifier/round_XX.pt         installed classifier heads
    policy_final.pt                final policy weights and normalization
    COMPLETE                       written last
While running, ``checkpoint/state.pt`` (policy, optimizer, reference, installed
head, round) and ``checkpoint/fit.pt`` (a classifier fit in progress) make the
run resumable; they are removed when the run completes.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from evenet_align.core.classifier import FeatureScale, FitState
from evenet_align.core.dgpo import (ClassifierRound, DGPOSetup, DGPOState, FinalPolicy, FitProgress,
                                    UpdateMetrics, ValidationDraw, train_dgpo)
from evenet_align.core.types import RunConfig

from . import factories
from .config import stage_hash, to_dict
from .datasets import load_events
from .policies import BACKENDS
from .rundir import final_dir, is_complete, prepare, provenance, publish, save_torch, write_json

STATE = "checkpoint/state.pt"
FIT = "checkpoint/fit.pt"


def _start(directory: Path, cfg: RunConfig, seed: int, rid: str, device: str, inputs: dict) -> None:
    (directory / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    write_json(directory / "provenance.json", {**provenance(to_dict(cfg), seed, rid, device),
                                               "inputs": inputs, "resumed_at_steps": []})


def _line(stage: str, step: int, values: dict) -> str:
    return json.dumps({"stage": stage, "step": step, **values}) + "\n"


def _state_to_disk(state: DGPOState, metrics_offset: int) -> dict:
    payload = asdict(state)
    payload["scale"] = None if state.scale is None else {"mean": state.scale.mean, "std": state.scale.std}
    return {"state": payload, "metrics_offset": metrics_offset}


def _load_resume(directory: Path, device: str) -> tuple[DGPOState, FitProgress | None]:
    saved = torch.load(directory / STATE, map_location=device, weights_only=True)
    payload = saved["state"]
    scale = payload.pop("scale")
    state = DGPOState(**payload, scale=None if scale is None else FeatureScale(**scale))
    with open(directory / "metrics.jsonl", "r+") as log:  # drop records written after the checkpoint
        log.truncate(saved["metrics_offset"])
    fit = None
    if (directory / FIT).is_file():
        saved_fit = torch.load(directory / FIT, map_location=device, weights_only=True)
        if saved_fit["step"] == state.step:
            fit = FitProgress(saved_fit["step"], FitState(**saved_fit["state"]))
    return state, fit


def run_dgpo(cfg: RunConfig, seed: int, results: Path, device: str) -> Path:
    backend = BACKENDS[cfg.policy.name]
    inputs = backend.inputs(cfg, seed)
    rid = f"dgpo-{stage_hash(cfg, 'dgpo', inputs)}-seed{seed}"
    final = final_dir(results, cfg.experiment, rid)
    if is_complete(final):
        return final
    train, validation = load_events(cfg, device)
    supervised = backend.load(cfg, seed, results, device, train)
    oracle = factories.build(cfg.oracle, train)
    extra_features = factories.build(cfg.classifier.features, train)

    directory, resuming = prepare(results, cfg.experiment, rid, checkpoint=STATE)
    resume = resume_fit = None
    if resuming:
        resume, resume_fit = _load_resume(directory, device)
        prov = json.loads((directory / "provenance.json").read_text())
        write_json(directory / "provenance.json", {**prov, "resumed_at_steps": prov["resumed_at_steps"] + [resume.step]})
    else:
        _start(directory, cfg, seed, rid, device, inputs)
        for sub in ("validation", "classifier", "checkpoint"):
            (directory / sub).mkdir()
        np.savez(directory / "validation" / "truth.npz", target=validation.target.cpu().numpy(),
                 event_id=validation.event_id.cpu().numpy())
        (directory / "metrics.jsonl").touch()

    oracle_constants = None
    with open(directory / "metrics.jsonl", "a") as log:
        for record in train_dgpo(supervised, train, validation, oracle, extra_features, cfg, seed, resume, resume_fit):
            if isinstance(record, DGPOSetup):
                oracle_constants = record.oracle_constants
            elif isinstance(record, DGPOState):
                log.flush()
                save_torch(_state_to_disk(record, log.tell()), directory / STATE)
                (directory / FIT).unlink(missing_ok=True)
            elif isinstance(record, FitProgress):
                save_torch({"step": record.step, "state": asdict(record.state)}, directory / FIT)
            elif isinstance(record, ValidationDraw):
                np.savez(directory / "validation" / f"step_{record.step:07d}.npz",
                         prediction=record.prediction.cpu().numpy())
                log.write(_line("validation", record.step, record.values))
            elif isinstance(record, ClassifierRound):
                torch.save(record.fit.head.state_dict(), directory / "classifier" / f"round_{record.round:02d}.pt")
                save_torch({"oracle": oracle_constants, "feature_mean": record.scale.mean,
                            "feature_std": record.scale.std}, directory / "setup.pt")
                log.write(_line("classifier", record.step, record.values))
            elif isinstance(record, UpdateMetrics):
                log.write(_line("dgpo", record.step, record.values))
            elif isinstance(record, FinalPolicy):
                torch.save(record.state, directory / "policy_final.pt")
    shutil.rmtree(directory / "checkpoint")
    return publish(directory, results, cfg.experiment, rid)


def run_all(cfg: RunConfig, results: Path, device: str) -> list[Path]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    return [run_dgpo(cfg, seed, results, device) for seed in cfg.seeds]
