"""Summaries and figures derived from completed run directories only.

Reads ``results/<experiment>/dgpo-*/`` (runs with a COMPLETE marker), never
imports the algorithm, and writes:

* ``history.csv``: per run and validation step, each metric and its percent
  change relative to the same run's step-0 (pretrained) draw.
* ``summary.csv``: per config and step, mean and sample std of the change over
  seeds, with the run count.
* ``history.png`` (if matplotlib is installed).

Metrics, all on one raw DDIM draw per validation event against the MC target:
``vote_distance_mean`` (the oracle distance, logged by the run),
``w1_obj<i>_f<j>`` 1D Wasserstein-1 per invisible object i and feature j,
``sw1_obj<i>`` sliced Wasserstein-1 per object (64 fixed projections), and
``rmse_obj<i>_f<j>`` per-event error.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """W1 between two equal-size samples."""
    return float(np.abs(np.sort(x) - np.sort(y)).mean())


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, projections: int = 64, seed: int = 0) -> float:
    directions = np.random.default_rng(seed).normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([wasserstein_1d(x @ d, y @ d) for d in directions]))


def draw_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    out = {}
    for i in range(target.shape[1]):
        out[f"sw1_obj{i}"] = sliced_wasserstein(prediction[:, i], target[:, i])
        for j in range(target.shape[2]):
            p, t = prediction[:, i, j], target[:, i, j]
            out[f"w1_obj{i}_f{j}"] = wasserstein_1d(p, t)
            out[f"rmse_obj{i}_f{j}"] = float(np.sqrt(np.mean((p - t) ** 2)))
    return out


def run_history(run: Path) -> list[dict]:
    target = np.load(run / "validation" / "truth.npz")["target"]
    logged = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["stage"] == "validation":
            logged[row["step"]] = row
    rows = []
    for path in sorted((run / "validation").glob("step_*.npz")):
        step = int(path.stem.split("_")[1])
        values = draw_metrics(np.load(path)["prediction"], target)
        values["vote_distance_mean"] = logged[step]["vote_distance_mean"]
        rows.append({"step": step, **values})
    baseline = rows[0]
    if baseline["step"] != 0:
        raise ValueError(f"{run}: no step-0 validation draw")
    history = []
    for row in rows:
        for metric, value in row.items():
            if metric == "step":
                continue
            change = 100.0 * (value / baseline[metric] - 1.0)
            history.append({"run": run.name, "step": row["step"], "metric": metric,
                            "value": value, "change_pct": change})
    return history


def completed_runs(experiment_dir: Path) -> list[Path]:
    return sorted(p for p in experiment_dir.glob("dgpo-*") if (p / "COMPLETE").is_file())


OWNER = ".evenet-align-summary"
PRODUCTS = {OWNER, "history.csv", "summary.csv", "history.png"}


def summarize(experiment_dir: Path, output: Path) -> Path:
    """Write the tables and figure into ``output``: a new directory, or one holding only an earlier summary."""
    runs = completed_runs(experiment_dir)
    if not runs:
        raise FileNotFoundError(f"no completed dgpo runs in {experiment_dir}")
    if output.exists() and not ((output / OWNER).is_file() and {p.name for p in output.iterdir()} <= PRODUCTS) \
            and any(output.iterdir()):
        raise FileExistsError(f"{output} holds other files; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    (output / OWNER).touch()
    history = [row for run in runs for row in run_history(run)]
    with open(output / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    groups = defaultdict(list)
    for row in history:
        config = row["run"].rsplit("-seed", 1)[0]
        groups[(config, row["step"], row["metric"])].append(row["change_pct"])
    summary = []
    for (config, step, metric), changes in sorted(groups.items()):
        n = len(changes)
        std = float(np.std(changes, ddof=1)) if n > 1 else math.nan
        summary.append({"config": config, "step": step, "metric": metric,
                        "change_pct_mean": float(np.mean(changes)), "change_pct_std": std, "n_runs": n})
    with open(output / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _plot(summary, output / "history.png")
    return output


def _plot(summary: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    shown = ["vote_distance_mean", "sw1_obj0", "sw1_obj1"]
    fig, axes = plt.subplots(1, len(shown), figsize=(180 / 25.4, 55 / 25.4), sharex=True)
    for ax, metric in zip(axes, shown):
        for config in sorted({r["config"] for r in summary}):
            rows = [r for r in summary if r["config"] == config and r["metric"] == metric]
            steps = np.array([r["step"] for r in rows])
            mean = np.array([r["change_pct_mean"] for r in rows])
            std = np.nan_to_num(np.array([r["change_pct_std"] for r in rows]))
            ax.plot(steps, mean, lw=1.2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.25, lw=0)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_title(metric.replace("_", " "), fontsize=8)
        ax.set_xlabel("policy updates", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("change vs pretrained [%]", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
