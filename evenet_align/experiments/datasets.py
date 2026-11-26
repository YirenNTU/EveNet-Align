"""Dataset boundary: build or read events and validate them once.

Sources are a registry keyed by ``dataset.source``.

``parquet``: EveNet preprocessed parquet directories, read directly (see ``convert.py``).

``npz``: ``<path>/train.npz`` and ``<path>/validation.npz``. Arrays named in
``dataset.observed`` become the policy inputs, ``dataset.target`` the MC
target (N, n, F), ``event_id`` (N,) and optional ``event_weight`` (N,) are
bookkeeping, and every other array goes to ``EventSet.extra`` for the oracle
and feature maps. If ``<target>_mask`` is present it must mark every
invisible object as valid. For EveNet, store arrays in the representation
EveNet trains on (the same values as its preprocessed inputs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from evenet_align.core.pseudo_data import pseudo_ttbar, pseudo_tau
from evenet_align.core.types import EventSet, RunConfig, derive_seed

from .convert import load_parquet_events
from .factories import resolve

BOOKKEEPING = ("event_id", "event_weight")


def _pseudo(cfg: RunConfig) -> tuple[EventSet, EventSet]:
    d = cfg.dataset
    make = {"tau": pseudo_tau, "ttbar": pseudo_ttbar}[d.pseudo]
    return (make(d.n_train, derive_seed(d.seed, "train"), 0),
            make(d.n_validation, derive_seed(d.seed, "validation"), d.n_train))


def _read_npz(path: Path, cfg: RunConfig) -> EventSet:
    with np.load(path) as f:
        return _from_arrays({key: np.asarray(f[key]) for key in f.files}, cfg, str(path))


def _from_arrays(raw: dict, cfg: RunConfig, path: str) -> EventSet:
    d = cfg.dataset
    arrays = {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in raw.items()}
    missing = ({*d.observed, d.target, "event_id"}) - set(arrays)
    if missing:
        raise KeyError(f"{path}: missing arrays {sorted(missing)}")
    mask_key = f"{d.target}_mask"
    if mask_key in arrays and not bool((arrays[mask_key] > 0).all()):
        raise ValueError(f"{path}: {mask_key} must mark every invisible object as valid")
    n = len(arrays["event_id"])
    observed = {key: arrays[key] if arrays[key].dtype == torch.bool or not arrays[key].is_floating_point()
                else arrays[key].float() for key in d.observed}
    reserved = {*d.observed, d.target, *BOOKKEEPING}
    extra = {key: value.float() if value.is_floating_point() else value
             for key, value in arrays.items() if key not in reserved}
    return EventSet(observed=observed, target=arrays[d.target].float(), event_id=arrays["event_id"].long(),
                    event_weight=arrays["event_weight"].float() if "event_weight" in arrays else torch.ones(n),
                    extra=extra)


def _npz(cfg: RunConfig) -> tuple[EventSet, EventSet]:
    root = Path(cfg.dataset.path)
    return _read_npz(root / "train.npz", cfg), _read_npz(root / "validation.npz", cfg)


def _parquet(cfg: RunConfig) -> tuple[EventSet, EventSet]:
    d = cfg.dataset
    derive = resolve(d.derive) if d.derive else None
    read = lambda directory, first_id: _from_arrays(load_parquet_events(
        Path(directory), keep={*d.observed, *d.extra}, target=d.target, first_id=first_id, derive=derive), cfg, directory)
    return read(d.train_parquet, 0), read(d.validation_parquet, 10**9)


SOURCES: dict[str, Callable[[RunConfig], tuple[EventSet, EventSet]]] = {
    "pseudo": _pseudo, "npz": _npz, "parquet": _parquet}


def validate(events: EventSet, name: str) -> None:
    """Reject malformed events instead of repairing them."""
    n = len(events)
    if events.target.ndim != 3:
        raise ValueError(f"{name}.target must have shape (N, n, F), got {tuple(events.target.shape)}")
    for group in ("observed", "extra"):
        for key, value in getattr(events, group).items():
            if value.shape[0] != n:
                raise ValueError(f"{name}.{group}[{key!r}] has {value.shape[0]} rows, expected {n}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name}.{group}[{key!r}] contains non-finite values")
    if not torch.isfinite(events.target).all():
        raise ValueError(f"{name}.target contains non-finite values")
    if tuple(events.event_id.shape) != (n,) or len(torch.unique(events.event_id)) != n:
        raise ValueError(f"{name}.event_id must be (N,) and unique")
    if tuple(events.event_weight.shape) != (n,) or not torch.isfinite(events.event_weight).all() \
            or (events.event_weight <= 0).any():
        raise ValueError(f"{name}.event_weight must be (N,), finite and positive")


def load_events(cfg: RunConfig, device: str) -> tuple[EventSet, EventSet]:
    train, validation = SOURCES[cfg.dataset.source](cfg)
    for events, name in ((train, "train"), (validation, "validation")):
        validate(events, name)
    if train.target.shape[1:] != validation.target.shape[1:]:
        raise ValueError("train and validation targets have different (n, F)")
    return train.to(device), validation.to(device)


def write_npz(events: EventSet, path: Path, target_key: str) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new output directory")
    arrays = {**events.observed, **events.extra, target_key: events.target, "event_id": events.event_id,
              "event_weight": events.event_weight}
    np.savez(path, **{k: v.cpu().numpy() for k, v in arrays.items()})
