"""Train: pretrain the baseline policy, then run DGPO, for every seed in the config.

    python scripts/run.py configs/smoke.yaml
    python scripts/run.py configs/ttbar.yaml --device cuda:0
"""
import argparse
import os
from pathlib import Path

# Bitwise-reproducible MKL kernels on CPU regardless of memory alignment; set before torch loads.
os.environ.setdefault("MKL_CBWR", "AUTO,STRICT")

from evenet_align.experiments.config import load_config
from evenet_align.experiments.run import run_all

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", type=Path)
parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
parser.add_argument("--results", type=Path, default=None, help="overrides the config's results")
parser.add_argument("--device", default=None, help="overrides the config's device, e.g. cuda:0")
args = parser.parse_args()
cfg = load_config(args.config, args.overrides)
for path in run_all(cfg, args.results or Path(cfg.results), args.device or cfg.device):
    print(path)
