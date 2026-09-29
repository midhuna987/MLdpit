#!/bin/bash
#==============================================================================================
# Required environment variables (export them, or give them on the sbatch line):
#   MLDPIT_ROOT_DIR  MLdpit repository root (the directory with pyproject.toml)
#   MLDPIT_VENV      Python virtual environment, absolute or relative to the root
#
# Usage:
#   cd /path/to/MLdpit && mkdir -p logs
#   MLDPIT_ROOT_DIR=$PWD MLDPIT_VENV=.venv \
#       sbatch --account=<your_project> slurm/train.sbatch
#
# Extra arguments are passed to mldpit-train and override configs/train.yaml,
# e.g. append --resume to continue from the last checkpoint.
#==============================================================================================
#SBATCH --job-name=mldpit-train
#SBATCH --partition=gpu
#SBATCH --nodes=12
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${MLDPIT_ROOT_DIR}"  

mkdir -p checkpoints

source "${MLDPIT_VENV:?set MLDPIT_VENV to the virtual environment, e.g. .venv}/bin/activate"


# Multi-node communication on DKRZ Levante.
# Native InfiniBand fails with the pip-installed NCCL 2.26.2
# ("Could not find NET with id 0"), so inter-node traffic runs
# as TCP over the InfiniBand interface ib0.

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     mldpit-train --config configs/train.yaml \
                  --strategy ddp --num-nodes "$SLURM_NNODES" \
                  --devices "$SLURM_NTASKS_PER_NODE" \
                  "$@"
                  



