"""Config boundary: YAML text and ``--set`` overrides -> typed ``RunConfig``.

Unknown keys and wrong types are rejected here; ``core`` trusts the result.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from pathlib import Path
from typing import Any

import yaml

from evenet_align.core.types import RunConfig


def _parse(cls: type, data: Any, where: str) -> Any:
    hints = typing.get_type_hints(cls)
    if not isinstance(data, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(data).__name__}")
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(hints)}")
    values = {}
    for name, value in data.items():
        values[name] = _coerce(hints[name], value, f"{where}.{name}")
    return cls(**values)


def _coerce(hint: Any, value: Any, where: str) -> Any:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if dataclasses.is_dataclass(hint):
        return _parse(hint, value, where)
    if origin is typing.Union:
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(inner, value, where)
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"{where}: {value!r} not one of {list(args)}")
        return value
    if hint is typing.Any:
        return value
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping")
        return dict(value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list")
        return tuple(_coerce(args[0], v, where) for v in value)
    if hint is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hint is bool and not isinstance(value, bool):
        raise TypeError(f"{where}: expected true/false, got {value!r}")
    if hint is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{where}: expected an integer, got {value!r}")
    if hint is str and not isinstance(value, str):
        raise TypeError(f"{where}: expected a string, got {value!r}")
    return value


def _apply_override(data: dict, assignment: str) -> None:
    key, sep, text = assignment.partition("=")
    if not sep:
        raise ValueError(f"override {assignment!r} must look like section.key=value")
    *parents, leaf = key.strip().split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(text)


def load_config(path: str | Path, overrides: list[str] = ()) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for assignment in overrides:
        _apply_override(data, assignment)
    cfg = _parse(RunConfig, data, "config")
    _validate(cfg)
    return cfg


def _validate(cfg: RunConfig) -> None:
    d, c, data, policy = cfg.dgpo, cfg.classifier, cfg.dataset, cfg.policy
    pseudo = data.source == "pseudo"
    checks = [
        (d.candidates >= 2, "dgpo.candidates must be >= 2"),
        (0.0 <= d.oracle_fraction < 1.0, "dgpo.oracle_fraction must lie in [0, 1)"),
        (0.0 <= d.t_min < d.t_max <= 1.0, "need 0 <= dgpo.t_min < dgpo.t_max <= 1"),
        (d.train_timesteps >= 1 and d.ddim_steps >= 1, "timesteps and DDIM steps must be positive"),
        (d.validate_every >= 1 and d.refit_every >= 1 and d.checkpoint_every >= 1,
         "validate_every, refit_every and checkpoint_every must be positive"),
        (0.0 < c.validation_fraction < 1.0, "classifier.validation_fraction must lie in (0, 1)"),
        (c.ratio_bound > 1.0, "classifier.ratio_bound must exceed 1"),
        (":" in cfg.oracle.function, "oracle.function must look like package.module:name"),
        (c.features is None or ":" in c.features.function, "classifier.features.function must look like package.module:name"),
        (not pseudo or c.fit_events is None or c.fit_events <= data.n_train,
         "classifier.fit_events exceeds dataset.n_train"),
        (not pseudo or d.batch_size <= data.n_train, "dgpo.batch_size exceeds dataset.n_train"),
        (data.source != "npz" or data.path, "dataset.path is required for source npz"),
        (data.source != "parquet" or (data.train_parquet and data.validation_parquet),
         "dataset.train_parquet and dataset.validation_parquet are required for source parquet"),
        (policy.name != "evenet" or all((policy.evenet.checkpoint, policy.evenet.config, policy.evenet.normalization)),
         "policy evenet needs policy.evenet.checkpoint, .config and .normalization from your evenet-train fine-tune"),
        (policy.name != "custom" or (policy.custom.function is not None and ":" in policy.custom.function),
         "policy custom needs policy.custom.function: package.module:name returning your Policy"),
        (len(cfg.seeds) >= 1, "seeds must list at least one seed"),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def to_dict(cfg: Any) -> Any:
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def stage_hash(cfg: RunConfig, stage: str, inputs: dict | None = None) -> str:
    """Hash of every setting that can change a stage's output (seeds and experiment name excluded).

    ``inputs`` adds digests of input files (e.g. the supervised checkpoint).
    """
    sections = {"supervised": ("dataset", "policy"),
                "dgpo": ("dataset", "policy", "oracle", "dgpo", "classifier", "validation")}[stage]
    payload = {name: to_dict(getattr(cfg, name)) for name in sections}
    payload["inputs"] = inputs or {}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
