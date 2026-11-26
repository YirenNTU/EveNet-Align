"""EveNet preprocessed parquet -> the npz layout DGPO reads.

EveNet stores each tensor as flat columns ``name:i:j`` and their shapes in
``shape_metadata.json`` next to the parquet files; scalar columns keep their
name. This reads every ``*.parquet`` in a directory (sorted), restores the
tensors, keeps events whose invisible objects are all valid (and
``event_valid`` > 0 when present), adds integer event ids, and calls an
optional ``derive(arrays) -> dict`` that computes the arrays your oracle needs
(``evenet_align.examples.ttbar:derive`` / ``...tau:derive``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def read_evenet_parquet(directory: Path, columns: Optional[set[str]] = None) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    shapes = json.loads((directory / "shape_metadata.json").read_text())
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {directory}")
    pieces: dict[str, list[np.ndarray]] = {}
    for path in files:
        source = pq.ParquetFile(path)
        names = [n for n in source.schema_arrow.names if columns is None or n.split(":")[0] in columns]
        table = source.read(columns=names)
        flat = {name: table.column(name).to_numpy(zero_copy_only=False) for name in names}
        grouped: dict[str, list[str]] = {}
        for name in flat:
            grouped.setdefault(name.split(":")[0], []).append(name)
        for base, cols in grouped.items():
            if base in shapes:
                cols = sorted(cols, key=lambda c: tuple(map(int, c.split(":")[1:])))
                value = np.stack([flat[c] for c in cols], 1).reshape(len(table), *shapes[base])
            else:
                value = flat[cols[0]]
            pieces.setdefault(base, []).append(value)
    return {k: np.concatenate(v) for k, v in pieces.items()}


def load_parquet_events(directory: Path, *, keep: set[str], target: str, first_id: int,
                        derive: Optional[Callable[[dict], dict]] = None) -> dict[str, np.ndarray]:
    """Valid events as the arrays DGPO reads: ``keep`` columns, target (+ mask), weights, derived extras, ids."""
    arrays = read_evenet_parquet(directory)
    if target not in arrays or f"{target}_mask" not in arrays:
        raise KeyError(f"{directory}: needs {target} and {target}_mask")
    valid = np.asarray(arrays[f"{target}_mask"]).astype(bool).all(1)
    if "event_valid" in arrays:
        valid &= np.asarray(arrays["event_valid"]).reshape(len(valid)) > 0
    arrays = {k: np.asarray(v)[valid] for k, v in arrays.items()}
    extra = derive(arrays) if derive is not None else {}
    out = {k: arrays[k] for k in keep | {target, f"{target}_mask"} if k in arrays}
    missing = (keep | {target}) - set(out)
    if missing:
        raise KeyError(f"{directory}: missing columns {sorted(missing)}")
    if "event_weight" in arrays:
        out["event_weight"] = np.asarray(arrays["event_weight"], dtype=np.float32).reshape(-1)
    out.update(extra)
    out["event_id"] = np.arange(int(valid.sum()), dtype=np.int64) + first_id
    return out


def convert(directory: Path, output: Path, *, keep: set[str], target: str, first_id: int,
            derive: Optional[Callable[[dict], dict]] = None) -> int:
    if output.exists():
        raise FileExistsError(f"{output} exists; converted data are never overwritten, choose a new --output")
    out = load_parquet_events(directory, keep=keep, target=target, first_id=first_id, derive=derive)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    return len(out["event_id"])
