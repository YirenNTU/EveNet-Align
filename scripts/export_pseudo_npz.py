"""Write the configured pseudo dataset as npz files, as a template for your own data.

    python scripts/export_pseudo_npz.py configs/smoke.yaml data/pseudo_tau
    python scripts/run.py configs/smoke.yaml --set dataset.source=npz --set dataset.path=data/pseudo_tau
"""
import argparse
from pathlib import Path

from evenet_align.experiments.config import load_config
from evenet_align.experiments.datasets import load_events, write_npz

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
args = parser.parse_args()
cfg = load_config(args.config, args.overrides)
train, validation = load_events(cfg, "cpu")
args.output.mkdir(parents=True, exist_ok=True)
write_npz(train, args.output / "train.npz", cfg.dataset.target)
write_npz(validation, args.output / "validation.npz", cfg.dataset.target)
print(args.output)
