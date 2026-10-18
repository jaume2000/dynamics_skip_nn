#!/bin/bash
#SBATCH --job-name=ft_s4_r100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/finetune_s4_r100_convnext_%j.out
#SBATCH --error=logs/finetune_s4_r100_convnext_%j.err

# Finetune ConvNeXt-T S4+head with stage-3 unrolled to R=100 (Δt=0.01, T=9).
# 10 epochs, LR=1e-6. Stem/S1/S2/S3 frozen.
#
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/finetune_s4_r100_convnext.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi

export REPEATS="${REPEATS:-100}"
export EULER_STEP="${EULER_STEP:-0.01}"
export EPOCHS="${EPOCHS:-10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export LR="${LR:-1e-6}"
export MIN_LR="${MIN_LR:-1e-8}"
export METHOD="${METHOD:-RK1}"
export WEIGHT_INTERPOLATION="${WEIGHT_INTERPOLATION:-plain}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-convnext_R100_ES0p01_finetune_s4}"

exec bash "${PROJECT_ROOT}/jobs/finetune_s4_r_convnext.sh" "$@"
