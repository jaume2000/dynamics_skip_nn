#!/bin/bash
#SBATCH --job-name=ft_s4_protocol
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/ft_s4_protocol_%j.out
#SBATCH --error=logs/ft_s4_protocol_%j.err

# Common S4+head finetune protocol (scripts/finetune_s4_head_protocol.py):
# frozen stem/S1-S3 in eval mode, constant lr 5e-6, 10 epochs, seed 0.
#
#   MODEL=untied REPEATS=1   sbatch --account="$SLURM_ACCOUNT" jobs/ft_s4_protocol.sh
#   MODEL=tied   REPEATS=1   sbatch ...
#   MODEL=untied REPEATS=3   sbatch ...
#   MODEL=untied REPEATS=100 sbatch ...   # or REPEATS=20, see the cost estimate
#   MODEL=tied   REPEATS=100 sbatch ...
#   FT_RETAKE=1 ... to resume from outputs/<FT_EXPERIMENT_NAME>/weights/last.pth
# The parent job sources .env: keep RETAKE=0 there, or its pre-flight check looks for
# outputs/$EXPERIMENT_NAME/weights/last.pth instead of the baseline weights.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi

export MODEL="${MODEL:-untied}"
export REPEATS="${REPEATS:-1}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-5e-6}"
export FT_EXPERIMENT_NAME="${FT_EXPERIMENT_NAME:-ft_s4_${MODEL}_R${REPEATS}_protocol}"
export FT_RETAKE="${FT_RETAKE:-0}"
if [[ "${MODEL}" == "tied" ]]; then
  export BASELINE_WEIGHTS="${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth"
else
  export BASELINE_WEIGHTS="${PROJECT_ROOT}/outputs/convnextv1_imagenet/weights/last.pth"
fi
export WARMUP_EPOCHS=0
export TRAIN_SCRIPT="scripts/finetune_s4_head_protocol.py"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

exec bash "${PROJECT_ROOT}/jobs/finetune_s4_r_convnext.sh" "$@"
