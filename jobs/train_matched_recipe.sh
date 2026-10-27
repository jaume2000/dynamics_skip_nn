#!/bin/bash
#SBATCH --job-name=matched
# Half booster node: 2 GPUs / 16 CPUs. Global batch 1024 via grad accum ×2.
# Chain 24 h segments with RETAKE=1 until 300 epochs finish.
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=250000
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/train_matched_%j.out
#SBATCH --error=logs/train_matched_%j.err

# Matched recipe (LR 1e-3, warmup 10, min 1e-7, B=1024) for tied or untied.
#
# Required env:
#   MODEL=tied|untied
#   SEED=<int>
#
# Optional:
#   MATCHED_EXPERIMENT_NAME=...   (default: shared_convnextv1_matched_seed{S}
#                                  or convnextv1_matched_seed{S})
#   RETAKE=0|1
#   NPROC_PER_NODE=2
#   STAGE_DATA=0   # skip /dev/shm staging (safer when packing several half-nodes)
#
# Submit the three planned runs (from repo root, after source .env):
#   MODEL=untied SEED=0  sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
#   MODEL=untied SEED=1  sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
#   MODEL=tied   SEED=1  sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
#
# Resume a timed-out segment:
#   MODEL=untied SEED=0 RETAKE=1 sbatch --account="$SLURM_ACCOUNT" --export=ALL \
#       jobs/train_matched_recipe.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RETAKE_CLI="${RETAKE:-}"
MODEL_CLI="${MODEL:-}"
SEED_CLI="${SEED:-}"
NAME_CLI="${MATCHED_EXPERIMENT_NAME:-}"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi
export RETAKE="${RETAKE_CLI:-${RETAKE:-0}}"
export MODEL="${MODEL_CLI:-${MODEL:-}}"
export SEED="${SEED_CLI:-${SEED:-}}"
if [[ -n "${NAME_CLI}" ]]; then
  export MATCHED_EXPERIMENT_NAME="${NAME_CLI}"
fi

if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required: tied | untied" >&2
  exit 1
fi
if [[ "${MODEL}" != "tied" && "${MODEL}" != "untied" ]]; then
  echo "MODEL must be tied|untied (got ${MODEL})" >&2
  exit 1
fi
if [[ -z "${SEED}" ]]; then
  echo "SEED is required (non-negative int)" >&2
  exit 1
fi

if [[ -z "${MATCHED_EXPERIMENT_NAME:-}" ]]; then
  if [[ "${MODEL}" == "tied" ]]; then
    MATCHED_EXPERIMENT_NAME="shared_convnextv1_matched_seed${SEED}"
  else
    MATCHED_EXPERIMENT_NAME="convnextv1_matched_seed${SEED}"
  fi
fi
export MATCHED_EXPERIMENT_NAME
export EXPERIMENT_NAME="${MATCHED_EXPERIMENT_NAME}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "SLURM_ACCOUNT is not set. Add it to ${PROJECT_ROOT}/.env" >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/logs"
  # --job-name includes model+seed so squeue is readable with 3 parallel jobs.
  exec sbatch \
    --account="${SLURM_ACCOUNT}" \
    --job-name="m_${MODEL}_s${SEED}" \
    --export=ALL \
    "${BASH_SOURCE[0]}" "$@"
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

EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

if [[ "${RETAKE}" == "1" ]]; then
  if [[ ! -f "${EXPERIMENT}/weights/last.pth" ]]; then
    echo "RETAKE=1 but checkpoint not found at ${EXPERIMENT}/weights/last.pth" >&2
    exit 1
  fi
fi

# Default STAGE_DATA=0 when packing several half-node jobs: three copies of
# ImageNet in /dev/shm will not fit. Set STAGE_DATA=1 to force staging.
if [[ "${STAGE_DATA:-0}" == "1" ]]; then
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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_matched_recipe.py}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "MODEL: ${MODEL}"
echo "SEED: ${SEED}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "Experiment path: ${EXPERIMENT}"
echo "RETAKE: ${RETAKE}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__}'); print(f'timm={timm.__version__}')"

export RETAKE MODEL SEED EXPERIMENT_NAME MATCHED_EXPERIMENT_NAME
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
