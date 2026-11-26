"""Convert EveNet preprocessed parquet directories (train and validation) into DGPO npz files.

    python scripts/evenet_parquet_to_npz.py --train <parquet dir> --validation <parquet dir> \
        --output data/ttbar --derive evenet_align.examples.ttbar:derive

Prints the number of events kept in each split.
"""
import argparse
from pathlib import Path

from evenet_align.experiments.convert import convert
from evenet_align.experiments.factories import resolve

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--train", type=Path, required=True)
parser.add_argument("--validation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--observed", nargs="+", default=["x", "x_mask", "conditions", "conditions_mask", "classification"])
parser.add_argument("--target", default="x_invisible")
parser.add_argument("--derive", default=None, help="package.module:function computing the oracle's extra arrays")
args = parser.parse_args()
derive = resolve(args.derive) if args.derive else None
n_train = convert(args.train, args.output / "train.npz", keep=set(args.observed), target=args.target,
                  first_id=0, derive=derive)
n_val = convert(args.validation, args.output / "validation.npz", keep=set(args.observed), target=args.target,
                first_id=10**9, derive=derive)
print(f"train {n_train} events, validation {n_val} events -> {args.output}")
