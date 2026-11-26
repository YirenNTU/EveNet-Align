#!/bin/bash
# Single-GPU DGPO run on NERSC Perlmutter. Arguments are passed to scripts/run.py;
# RESULTS must name a new results folder (or one this package created):
#   RESULTS=$SCRATCH/my-run/results sbatch -A <account> scripts/perlmutter.sbatch configs/ztautau.yaml --set ...
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node 1
#SBATCH -t 12:00:00
#SBATCH --requeue
set -euo pipefail
# A requeued job resumes unfinished runs from their last checkpoint and skips finished ones.
srun python scripts/run.py "$@" --device cuda --results "${RESULTS:?set RESULTS to a new results folder}"
