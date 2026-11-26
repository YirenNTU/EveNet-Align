"""Summarize completed DGPO runs of one experiment into CSV tables and a figure.

    python scripts/summarize.py results/smoke --output figures/smoke
"""
import argparse
from pathlib import Path

from evenet_align.analysis.summarize import summarize

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("experiment_dir", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
print(summarize(args.experiment_dir, args.output))
