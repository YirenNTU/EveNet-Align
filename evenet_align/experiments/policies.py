"""Policy backends: where the supervised starting policy of RL comes from.

* ``evenet`` (default): a checkpoint you fine-tuned with ``evenet-train``
  from the released pretrained weights. The released pretrained files
  themselves are refused: RL starts after supervised learning.
* ``mlp``: the small reference policy, trained here by its own supervised run
  (``supervised-<hash>-seed<s>``). For tests and pseudo data.
* ``custom``: your own supervised model, built by ``policy.custom.function``
  (``package.module:name``) as ``name(train_events, **params)``. It must be an
  ``nn.Module`` implementing ``core.types.Policy``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import yaml

from evenet_align.core.evenet_policy import EveNetPolicy
from evenet_align.core.models import build_mlp_policy, mlp_statistics
from evenet_align.core.supervised import SupervisedMetrics, SupervisedPolicy, train_supervised
from evenet_align.core.types import EventSet, Policy, RunConfig, derive_seed

from .config import stage_hash, to_dict
from .factories import resolve
from .rundir import final_dir, is_complete, prepare, provenance, publish, write_json

# sha256 of the released pretrained EveNet weights (huggingface.co/Avencast/EveNet)
RELEASED_PRETRAINED = {
    "e2ed50ca4f2c44fc1692c07c504ffdbad7d4a2c36a8d6206b6b3e2a7f5f148b2": "SSL.20M.last.ckpt",
    "3c8b571328112988fd7186953a2f3155f0e22301a9c8d7e24f520881251a5ef2": "checkpoints.20M.a4.last.ckpt",
}
HEADS = {"classification": "Classification", "regression": "Regression", "global_generation": "GlobalGeneration",
         "point_cloud_generation": "ReconGeneration", "neutrino_generation": "TruthGeneration",
         "assignment": "Assignment", "segmentation": "Segmentation"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# EveNet
# ----------------------------------------------------------------------------


def _evenet_inputs(cfg: RunConfig, seed: int) -> dict:
    e = cfg.policy.evenet
    for name in ("checkpoint", "config", "normalization"):
        if not Path(getattr(e, name)).is_file():
            raise FileNotFoundError(f"policy.evenet.{name}: {getattr(e, name)} does not exist")
    digest = sha256(e.checkpoint)
    if digest in RELEASED_PRETRAINED:
        raise ValueError(
            f"{e.checkpoint} is the released pretrained EveNet ({RELEASED_PRETRAINED[digest]}). RL starts from a "
            "supervised policy: fine-tune it on your events with evenet-train "
            "(options.Training.pretrain_model_load_path set to this file, TruthGeneration included) and set "
            "policy.evenet.checkpoint to the fine-tuned checkpoint.")
    return {"evenet_checkpoint_sha256": digest, "evenet_config_sha256": sha256(e.config),
            "evenet_normalization_sha256": sha256(e.normalization), "evenet_source": _source_revision(e.source)}


def _source_revision(source: str | None) -> str | None:
    """git commit (and dirty flag) of a custom EveNet checkout, for provenance and the run hash."""
    if source is None:
        return None
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=source, capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "HEAD")
    return f"{commit}{'+dirty' if run('status', '--porcelain') else ''}" if commit else str(Path(source).resolve())


def _evenet_load(cfg: RunConfig, seed: int, results: Path, device: str, train: EventSet) -> Policy:
    e = cfg.policy.evenet
    if e.source is not None:  # the EveNet code that trained this checkpoint takes precedence
        sys.path.insert(0, str(Path(e.source).resolve()))
    try:
        from evenet.control.global_config import Config
        from evenet.network.evenet_model import EveNetModel
    except ImportError as error:
        raise ImportError("policy evenet needs the EveNet package (pip install evenet-core) "
                          "or policy.evenet.source pointing to an EveNet checkout") from error
    config = Config()
    config.load_yaml(e.config)
    components = config.options.Training.Components
    heads = {flag: bool(components.get(name, {}).get("include", False)) for flag, name in HEADS.items()}
    if not heads["neutrino_generation"]:
        raise ValueError(f"{e.config}: options.Training.Components.TruthGeneration.include must be true")
    normalization = torch.load(e.normalization, map_location=device, weights_only=False)
    if e.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(e.float32_matmul_precision)

    def build() -> torch.nn.Module:
        model = EveNetModel(config=config, device=torch.device(device), normalization_dict=normalization, **heads)
        _configure_memory(model, e)
        return model.to(device)

    policy = EveNetPolicy(build, tuple(train.target.shape[1:]))
    # A Lightning checkpoint written by evenet-train; it is your own file. As EveNet does, use the
    # EMA weights only when EMA is enabled with replace_model_after_load.
    checkpoint = torch.load(e.checkpoint, map_location=device, weights_only=False)
    ema = config.options.Training.get("EMA", {}) or {}
    use_ema = bool(ema.get("enable", False) and ema.get("replace_model_after_load", False)
                   and "ema_state_dict" in checkpoint)
    raw = checkpoint["ema_state_dict" if use_ema else "state_dict"]
    state = {k.removeprefix("model."): v for k, v in raw.items()}
    # Loss-balancing (FAMO) parameters live beside the network in some checkpoints; they are not the policy.
    state = {k: v for k, v in state.items() if not k.startswith("famo.")}
    missing, unexpected = policy.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"{e.checkpoint} does not match the model built from {e.config}: "
                         f"missing {missing[:5]}, unexpected {unexpected[:5]} (set policy.evenet.source to the "
                         "EveNet code that trained it)")
    probe = policy.normalize(train.target[:2])
    if probe.shape[1] != train.target[0].numel():
        raise ValueError("EveNet invisible features do not match dataset.target (n, F)")
    return policy


def _configure_memory(model: torch.nn.Module, e) -> None:
    """Activation checkpointing / memory mode through EveNet's own utilities, as the research trainer did.

    Flags only change how the backward pass stores activations; parameters and
    forward results are unchanged. They take effect only in training mode with
    gradients, so frozen copies are unaffected.
    """
    if e.activation_checkpointing:
        try:
            from evenet.utilities.partial_checkpointing import configure_transformer_checkpointing
        except ImportError as error:
            raise ImportError("activation_checkpointing needs an EveNet build with "
                              "evenet.utilities.partial_checkpointing (set policy.evenet.source)") from error
        if configure_transformer_checkpointing(model, enabled=True, retain_last_blocks=0) == 0:
            raise ValueError("activation_checkpointing: the model has no checkpointable PET body")
    if e.activation_memory_mode is not None:
        try:
            from evenet.utilities.activation_offload import configure_activation_offload
        except ImportError as error:
            raise ImportError("activation_memory_mode needs an EveNet build with "
                              "evenet.utilities.activation_offload (set policy.evenet.source)") from error
        configure_activation_offload(model, budget_mib=0, mode=e.activation_memory_mode)


# ----------------------------------------------------------------------------
# Reference MLP with its own supervised run
# ----------------------------------------------------------------------------


def supervised_run_id(cfg: RunConfig, seed: int) -> str:
    return f"supervised-{stage_hash(cfg, 'supervised')}-seed{seed}"


def _mlp_inputs(cfg: RunConfig, seed: int) -> dict:
    return {"supervised_run": supervised_run_id(cfg, seed)}


def _mlp_build(cfg: RunConfig, train: EventSet, stats: dict, seed: int):
    return build_mlp_policy(train.observed["condition"].shape[1], tuple(train.target.shape[1:]), cfg.policy.mlp,
                            stats, seed)


def run_supervised(cfg: RunConfig, seed: int, results: Path, device: str, train: EventSet) -> Path:
    """Supervised velocity-MSE run of the MLP; redone from scratch if interrupted."""
    rid = supervised_run_id(cfg, seed)
    final = final_dir(results, cfg.experiment, rid)
    if is_complete(final):
        return final
    policy = _mlp_build(cfg, train, mlp_statistics(train), derive_seed(seed, "policy-init")).to(device)
    directory, _ = prepare(results, cfg.experiment, rid, checkpoint=None)
    (directory / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    write_json(directory / "provenance.json", provenance(to_dict(cfg), seed, rid, device))
    with open(directory / "metrics.jsonl", "w") as log:
        for record in train_supervised(policy, train, cfg.policy.mlp, seed):
            if isinstance(record, SupervisedPolicy):
                torch.save(record.state, directory / "policy.pt")
            elif isinstance(record, SupervisedMetrics):
                log.write(json.dumps({"stage": "supervised", "step": record.step, **record.values}) + "\n")
    return publish(directory, results, cfg.experiment, rid)


def _mlp_load(cfg: RunConfig, seed: int, results: Path, device: str, train: EventSet) -> Policy:
    run = run_supervised(cfg, seed, results, device, train)
    state = torch.load(run / "policy.pt", map_location=device, weights_only=True)
    policy = _mlp_build(cfg, train, {k: state[k] for k in ("condition_mean", "condition_std", "target_mean",
                                                            "target_std")}, 0).to(device)
    policy.load_state_dict(state)
    return policy


# ----------------------------------------------------------------------------
# Your own policy
# ----------------------------------------------------------------------------

POLICY_INTERFACE = ("normalize", "denormalize", "velocity", "condition_embedding", "candidate_features")


def _custom_inputs(cfg: RunConfig, seed: int) -> dict:
    return {"custom_files_sha256": {f: sha256(f) for f in cfg.policy.custom.files}}


def _custom_load(cfg: RunConfig, seed: int, results: Path, device: str, train: EventSet) -> Policy:
    c = cfg.policy.custom
    policy = resolve(c.function)(train, **c.params)
    if not isinstance(policy, torch.nn.Module):
        raise TypeError(f"{c.function} must return an nn.Module implementing Policy, got {type(policy).__name__}")
    missing = [name for name in ("target_shape", *POLICY_INTERFACE) if not hasattr(policy, name)]
    if missing:
        raise TypeError(f"{c.function} returned a policy without {missing} (see evenet_align.core.types.Policy)")
    if tuple(policy.target_shape) != tuple(train.target.shape[1:]):
        raise ValueError(f"policy target_shape {tuple(policy.target_shape)} does not match dataset.target "
                         f"{tuple(train.target.shape[1:])}")
    return policy.to(device)


@dataclass(frozen=True)
class Backend:
    inputs: Callable[[RunConfig, int], dict]
    load: Callable[[RunConfig, int, Path, str, EventSet], Policy]


BACKENDS: dict[str, Backend] = {"evenet": Backend(_evenet_inputs, _evenet_load), "mlp": Backend(_mlp_inputs, _mlp_load),
                                "custom": Backend(_custom_inputs, _custom_load)}
