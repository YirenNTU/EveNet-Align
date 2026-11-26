"""EveNet parquet -> npz conversion, including the ttbar and tau derived arrays."""
import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from evenet_align.examples import tau, ttbar  # noqa: E402
from evenet_align.experiments.convert import convert  # noqa: E402


def _write(directory, arrays, shapes, files=2):
    directory.mkdir(parents=True)
    (directory / "shape_metadata.json").write_text(json.dumps(shapes))
    n = len(next(iter(arrays.values())))
    for f, rows in enumerate(np.array_split(np.arange(n), files)):
        columns = {}
        for name, value in arrays.items():
            value = value[rows]
            if name in shapes:
                for index in np.ndindex(*shapes[name]):
                    columns[":".join([name, *map(str, index)])] = value[(slice(None), *index)]
            else:
                columns[name] = value
        pq.write_table(pa.table(columns), directory / f"part-{f}.parquet")


def test_round_trip_with_ttbar_derive(tmp_path):
    g = np.random.default_rng(0)
    n = 50
    arrays = {"x": g.normal(size=(n, 12, 7)).astype(np.float32), "x_mask": g.random((n, 12)) > 0.3,
              "conditions": g.normal(size=(n, 10)).astype(np.float32), "conditions_mask": np.ones((n, 1), bool),
              "classification": np.zeros(n, np.int64), "x_invisible": g.normal(size=(n, 2, 3)).astype(np.float32),
              "x_invisible_mask": np.ones((n, 2), bool), "event_valid": (np.arange(n) % 10 != 0).astype(np.float32)}
    arrays["x_invisible_mask"][3, 1] = False
    for i in (1, 2):
        for obj in ("b", "l"):
            for v, lo, hi in (("pt", 20, 200), ("eta", -2, 2), ("phi", -3, 3), ("energy", 30, 400)):
                arrays[f"EXTRA/t{i}/{obj}/{v}"] = g.uniform(lo, hi, n).astype(np.float32)
    shapes = {"x": [12, 7], "x_mask": [12], "conditions": [10], "conditions_mask": [1], "x_invisible": [2, 3],
              "x_invisible_mask": [2]}
    _write(tmp_path / "raw", arrays, shapes)
    kept = convert(tmp_path / "raw", tmp_path / "out.npz", keep={"x", "x_mask", "conditions", "conditions_mask",
                   "classification"}, target="x_invisible", first_id=7, derive=ttbar.derive)
    valid = arrays["x_invisible_mask"].all(1) & (arrays["event_valid"] > 0)
    assert kept == valid.sum()
    out = np.load(tmp_path / "out.npz")
    np.testing.assert_array_equal(out["x"], arrays["x"][valid])
    np.testing.assert_array_equal(out["x_invisible"], arrays["x_invisible"][valid])
    np.testing.assert_array_equal(out["classification"], arrays["classification"][valid])
    np.testing.assert_array_equal(out["event_id"], np.arange(kept) + 7)
    b = {v: arrays[f"EXTRA/t1/b/{v}"][valid].astype(np.float64) for v in ("pt", "eta", "phi", "energy")}
    l = {v: arrays[f"EXTRA/t1/l/{v}"][valid].astype(np.float64) for v in ("pt", "eta", "phi", "energy")}
    np.testing.assert_allclose(out["visible"][:, 0, 0], b["energy"] + l["energy"], rtol=1e-6)
    np.testing.assert_allclose(out["visible"][:, 0, 1], b["pt"] * np.cos(b["phi"]) + l["pt"] * np.cos(l["phi"]),
                               rtol=1e-5, atol=1e-3)
    assert out["visible"].shape == (kept, 2, 4)


def test_tau_derive_shapes():
    n = 4
    arrays = {f"lead_{leg}_visible_{q}": np.ones(n) for leg in "ab" for q in ("px", "py", "pz")}
    arrays.update({f"truth_tau_{leg}_{q}": np.full(n, 2.0) for leg in "ab" for q in ("E", "px", "py", "pz")})
    out = tau.derive(arrays)
    assert out["visible"].shape == (n, 2, 3) and out["truth_p4"].shape == (n, 2, 4)


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "out.npz"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        convert(tmp_path, target, keep=set(), target="x_invisible", first_id=0)
    assert target.read_bytes() == b"original"


def test_parquet_source_reads_the_same_events(tmp_path):
    from pathlib import Path

    from evenet_align.core.pseudo_data import pseudo_tau
    from evenet_align.experiments.config import load_config
    from evenet_align.experiments.datasets import load_events

    sets = {"train": pseudo_tau(300, 1), "validation": pseudo_tau(100, 2)}
    for split, events in sets.items():
        arrays = {"condition": events.observed["condition"].numpy(), "target": events.target.numpy(),
                  "target_mask": np.ones((len(events), 2), bool), "visible": events.extra["visible"].numpy(),
                  "truth_p4": events.extra["truth_p4"].numpy()}
        shapes = {"condition": [8], "target": [2, 2], "target_mask": [2], "visible": [2, 3], "truth_p4": [2, 4]}
        _write(tmp_path / split, arrays, shapes)
    smoke = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    cfg = load_config(smoke, ["dataset.source=parquet", f"dataset.train_parquet={tmp_path / 'train'}",
                              f"dataset.validation_parquet={tmp_path / 'validation'}",
                              "dataset.extra=[visible,truth_p4]"])
    train, validation = load_events(cfg, "cpu")
    for loaded, split in ((train, "train"), (validation, "validation")):
        original = sets[split]
        np.testing.assert_array_equal(loaded.target.numpy(), original.target.numpy())
        np.testing.assert_array_equal(loaded.observed["condition"].numpy(), original.observed["condition"].numpy())
        np.testing.assert_array_equal(loaded.extra["truth_p4"].numpy(), original.extra["truth_p4"].numpy())
