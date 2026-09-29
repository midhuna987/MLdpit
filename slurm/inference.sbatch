#!/bin/bash
#==============================================================================
# Required environment variables (export them, or give them on the sbatch line):
#   MLDPIT_ROOT_DIR  MLdpit repository root (the directory with pyproject.toml)
#   MLDPIT_VENV      Python virtual environment, absolute or relative to the root
#
# Usage:
#   cd /path/to/MLdpit && mkdir -p logs
#   MLDPIT_ROOT_DIR=$PWD MLDPIT_VENV=.venv \
#       sbatch --account=<your_project> slurm/inference.sbatch
#
# Extra arguments are passed to mldpit-infer and override
# configs/inference.yaml, e.g. a two-day test run:
#   ... slurm/inference.sbatch --period-end 2020-01-03 --output output/test.nc
#
# The output format follows the --output extension: .nc gives NetCDF,
# .zarr gives Zarr. An existing output at the same path is overwritten.
#==============================================================================
#SBATCH --job-name=mldpit-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$MLDPIT_ROOT_DIR" 

source "${MLDPIT_VENV}/bin/activate"


srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     mldpit-infer --config configs/inference.yaml "$@"


                  

