#!/bin/bash
#SBATCH --job-name=delta_feature_maps
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/delta_feature_maps_%j.out
#SBATCH --error=logs/delta_feature_maps_%j.err

# Stage-3 trajectories of the four delta ConvNeXt runs (R1 and R100, ± ignore).
# Missing checkpoints are skipped so you can submit as each train job finishes.
#
#   source .env && sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/delta_feature_maps.sh
#   TRAIN_ARGS='--only delta_d9_lowlr_baseline_R1_ES1_c289_n1' ...

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
  exec sbatch --account="${SLURM_ACCOUNT}" --export=ALL "${BASH_SOURCE[0]}" "$@"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

module purge
module load profile/deeplrn
module load cineca-ai
module load ffmpeg 2>/dev/null || true

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
export MPLBACKEND=Agg
export FEATURE_MAP_BACKBONE=delta

HF_HOME="${HF_HOME:-${WORK:+$WORK/huggingface}}"
HF_HOME="${HF_HOME:-${CINECA_SCRATCH:-$HOME}/hf}"
HF_HOME="${HF_HOME/#\~/$HOME}"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -z "${FEATURE_MAP_ROOT:-}" ]]; then
  if [[ -n "${WORK:-}" ]]; then
    FEATURE_MAP_ROOT="${WORK}/feature_maps"
  else
    FEATURE_MAP_ROOT="${PROJECT_ROOT}/outputs"
  fi
fi
FEATURE_MAP_ROOT="${FEATURE_MAP_ROOT/#\~/$HOME}"
export FEATURE_MAP_ROOT
mkdir -p "${FEATURE_MAP_ROOT}/featureMaps" "${FEATURE_MAP_ROOT}/featureMaps_interpoled"

DATASET_DIR="ILSVRC___imagenet-1k"
if [[ ! -d "${HF_DATASETS_CACHE}/${DATASET_DIR}" ]]; then
  echo "ImageNet cache not found at ${HF_DATASETS_CACHE}/${DATASET_DIR}" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "FEATURE_MAP_ROOT: ${FEATURE_MAP_ROOT}"
echo "FEATURE_MAP_BACKBONE: ${FEATURE_MAP_BACKBONE}"
echo "Args: ${TRAIN_ARGS:-}"
echo "Start: $(date)"

# shellcheck disable=SC2086
python "${PROJECT_ROOT}/scripts/feature_map_explorer.py" ${TRAIN_ARGS:-}

echo "End: $(date)"
