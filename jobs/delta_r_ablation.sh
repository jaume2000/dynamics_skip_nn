#!/bin/bash
#SBATCH --job-name=delta_r_ablation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/delta_r_ablation_%j.out
#SBATCH --error=logs/delta_r_ablation_%j.err

# Classic R-refine ablation (R=1..128 × RK1/2/4, ES=1/R) for one delta run.
# Set DELTA_EXPERIMENT to the outputs/ folder name:
#   DELTA_EXPERIMENT=delta_from_D9_lr1e-5_e100 source .env && \
#     sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/delta_r_ablation.sh
#
# Or run every finished variant:
#   DELTA_EXPERIMENT=all source .env && sbatch --account="$SLURM_ACCOUNT" \
#       --export=ALL jobs/delta_r_ablation.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

EXPERIMENT_CLI="${DELTA_EXPERIMENT:-}"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi
export DELTA_EXPERIMENT="${EXPERIMENT_CLI:-${DELTA_EXPERIMENT:-}}"

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

ALL_EXPERIMENTS=(
  delta_from_D9_lr1e-5_e100
  delta_from_D9_ramp0to1_e80
  delta_scratch_ramp150to250_e300
  delta_scratch_full_e300
)

if [[ -z "${DELTA_EXPERIMENT}" ]]; then
  echo "DELTA_EXPERIMENT is required (a run name, or 'all')" >&2
  exit 1
fi

if [[ "${DELTA_EXPERIMENT}" == "all" ]]; then
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
else
  EXPERIMENTS=("${DELTA_EXPERIMENT}")
fi

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Start: $(date)"

ran=0
for name in "${EXPERIMENTS[@]}"; do
  ckpt="${PROJECT_ROOT}/outputs/${name}/weights/last.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skip ${name}: no checkpoint at ${ckpt}" >&2
    continue
  fi
  echo "=== R ablation ${name} ==="
  python "${PROJECT_ROOT}/scripts/delta_convnext_interpolation.py" --experiment "${name}"
  ran=$((ran + 1))
done

python "${PROJECT_ROOT}/scripts/plot_delta_experiments.py" || true

if (( ran == 0 )); then
  echo "No delta checkpoints were ready." >&2
  exit 1
fi

echo "End: $(date)"
