#!/bin/bash
#SBATCH --job-name=ft_s4_d900
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=250000
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/finetune_s4_d900_shared_%j.out
#SBATCH --error=logs/finetune_s4_d900_shared_%j.err

# Finetune shared ConvNeXt-T S4+head with stage-3 unrolled to D=900 (Δt=0.01, T=9; same step as R=100).
# 10 epochs, constant LR=5e-6. Stem/S1/S2/S3 frozen.
#
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/finetune_s4_d900_shared_convnext.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "SLURM_ACCOUNT is not set. Add it to ${PROJECT_ROOT}/.env" >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/logs"
  exec sbatch --account="${SLURM_ACCOUNT}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

module purge
module load profile/deeplrn
module load cineca-ai

VENV_PATH="${VENV_PATH:-$HOME/venvs/convnext}"
VENV_PATH="${VENV_PATH/#\~/$HOME}"
PY_TAG="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_SITE="${VENV_PATH}/lib/python${PY_TAG}/site-packages"
if [[ ! -d "${VENV_SITE}" ]]; then
  echo "Venv site-packages not found at '${VENV_SITE}'." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

HF_HOME="${HF_HOME:-${WORK:+$WORK/huggingface}}"
HF_HOME="${HF_HOME:-${CINECA_SCRATCH:-$HOME}/hf}"
HF_HOME="${HF_HOME/#\~/$HOME}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  exit 1
fi

export D="${D:-900}"
export EULER_STEP="${EULER_STEP:-0.01}"
export EPOCHS="${EPOCHS:-10}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
export LR="${LR:-5e-6}"
export MIN_LR="${MIN_LR:-5e-6}"
export METHOD="${METHOD:-RK1}"
export RETAKE="${RETAKE:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-convnext_shared_D900_ES0p01_finetune_s4}"
EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/finetune_s4_d_shared_convnext.py}"

if [[ "${RETAKE}" == "1" ]]; then
  if [[ ! -f "${EXPERIMENT}/weights/last.pth" ]]; then
    echo "RETAKE=1 but checkpoint not found at ${EXPERIMENT}/weights/last.pth" >&2
    exit 1
  fi
else
  if [[ ! -f "${BASELINE_WEIGHTS}" ]]; then
    echo "Shared baseline weights not found at ${BASELINE_WEIGHTS}" >&2
    exit 1
  fi
fi

if [[ "${STAGE_DATA:-1}" == "1" ]]; then
  STAGE_ROOT="/dev/shm/hfstage_${SLURM_JOB_ID}"
  needed_kb="$(du -sk "${HF_DATASETS_CACHE}/${DATASET_DIR}" | cut -f1)"
  avail_kb="$(df --output=avail -k /dev/shm | tail -1)"
  margin_kb=$((30 * 1024 * 1024))
  if (( needed_kb + margin_kb > avail_kb )); then
    echo "Skipping staging: need $((needed_kb / 1024 / 1024))G + 30G margin," \
         "/dev/shm has $((avail_kb / 1024 / 1024))G" >&2
  else
    trap 'rm -rf "${STAGE_ROOT}"' EXIT TERM INT
    echo "Staging $((needed_kb / 1024 / 1024))G into ${STAGE_ROOT}"
    stage_start="${SECONDS}"
    pushd "${HF_DATASETS_CACHE}" > /dev/null
    find "${DATASET_DIR}" -type d -exec mkdir -p "${STAGE_ROOT}/{}" \;
    find "${DATASET_DIR}" -type f -print0 | xargs -0 -P 8 -I{} cp -p {} "${STAGE_ROOT}/{}"
    popd > /dev/null
    echo "Staged in $((SECONDS - stage_start))s"
    export HF_DATASETS_CACHE="${STAGE_ROOT}"
  fi
fi

echo "Host: $(hostname)"
echo "D=${D} EULER_STEP=${EULER_STEP} EPOCHS=${EPOCHS} LR=${LR}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "Baseline: ${BASELINE_WEIGHTS}"
echo "Start: $(date)"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
