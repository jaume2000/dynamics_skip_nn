#!/bin/bash
#SBATCH --job-name=delta_train
# Short segments schedule sooner than the 4-day cap; chain with RETAKE=1.
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Full booster node (32 cores / 4 GPUs). Finetunes use global batch 512
# (grad accum if needed); from-scratch uses global batch 1024.
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/delta_train_%j.out
#SBATCH --error=logs/delta_train_%j.err

# Four variants (VARIANT is required):
#   d9_lowlr       D=9 shared → deltas, LR=1e-5, 100 epochs
#   d9_ramp        D=9 shared → deltas, coeff 0→1 by epoch 60, end 80, LR=1e-4
#   scratch_ramp   from scratch, coeff 0 until 150, 1 at 250, end 300
#   scratch_full   from scratch, coeff always 1, 300 epochs
#
# Submit from the repo root (new Leonardo account: AIFAC_P02_1564):
#   VARIANT=d9_lowlr source .env && sbatch --account="$SLURM_ACCOUNT" \
#       --export=ALL jobs/train_delta_experiment.sh
#
# Chain after a time-limit kill:
#   VARIANT=d9_lowlr RETAKE=1 source .env && sbatch --account="$SLURM_ACCOUNT" \
#       --export=ALL jobs/train_delta_experiment.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RETAKE_CLI="${RETAKE:-}"
VARIANT_CLI="${VARIANT:-}"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi
export RETAKE="${RETAKE_CLI:-${RETAKE:-0}}"
export VARIANT="${VARIANT_CLI:-${VARIANT:-}}"

if [[ -z "${VARIANT}" ]]; then
  echo "VARIANT is required: d9_lowlr | d9_ramp | scratch_ramp | scratch_full" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "SLURM_ACCOUNT is not set. Add it to ${PROJECT_ROOT}/.env" >&2
    echo "For the new project use SLURM_ACCOUNT=AIFAC_P02_1564" >&2
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
  echo "Venv site-packages not found at '${VENV_SITE}'. Create it on a login node:" >&2
  echo "  module load profile/deeplrn && module load cineca-ai" >&2
  echo "  python -m venv --system-site-packages \"\$HOME/venvs/convnext\"" >&2
  echo "  source \"\$HOME/venvs/convnext/bin/activate\" && pip install --force-reinstall --no-deps timm" >&2
  echo "  deactivate && export PYTHONNOUSERSITE=1" >&2
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
  echo "Set HF_HOME / HF_DATASETS_CACHE in .env to the folder that contains ${DATASET_DIR}." >&2
  exit 1
fi

# Forced from VARIANT so .env's EXPERIMENT_NAME (the shared baseline) cannot win.
case "${VARIANT}" in
  d9_lowlr) EXPERIMENT_NAME="delta_from_D9_lr1e-5_e100" ;;
  d9_ramp) EXPERIMENT_NAME="delta_from_D9_ramp0to1_e80" ;;
  scratch_ramp) EXPERIMENT_NAME="delta_scratch_ramp150to250_e300" ;;
  scratch_full) EXPERIMENT_NAME="delta_scratch_full_e300" ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 1
    ;;
esac
EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
PRETRAINED="${PROJECT_ROOT}/outputs/shared_convnextv1_imagenet/weights/last.pth"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ "${RETAKE}" == "1" ]]; then
  if [[ ! -f "${EXPERIMENT}/weights/last.pth" ]]; then
    echo "RETAKE=1 but checkpoint not found at ${EXPERIMENT}/weights/last.pth" >&2
    echo "The first segment of this experiment must run with RETAKE=0." >&2
    exit 1
  fi
elif [[ "${VARIANT}" == d9_* && ! -f "${PRETRAINED}" ]]; then
  echo "Pretrained shared-only checkpoint not found at ${PRETRAINED}" >&2
  echo "outputs/ is gitignored; copy it from the previous project (797 → 1564)." >&2
  exit 1
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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_delta_experiment.py}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Venv site: ${VENV_SITE}"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "VARIANT: ${VARIANT}"
echo "RETAKE: ${RETAKE}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "Experiment path: ${EXPERIMENT}"
echo "Pretrained: ${PRETRAINED} ($([ -f "${PRETRAINED}" ] && echo ok || echo missing))"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})'); print(f'timm={timm.__version__} ({timm.__file__})')"

export RETAKE
export VARIANT
export EXPERIMENT_NAME
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
