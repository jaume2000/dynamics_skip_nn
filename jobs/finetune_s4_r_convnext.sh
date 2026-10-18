#!/bin/bash
#SBATCH --job-name=ft_s4_r3
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Booster nodes are 32 cores / 4 GPUs; asking for fewer just idles dataloader cores.
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/finetune_s4_r_convnext_%j.out
#SBATCH --error=logs/finetune_s4_r_convnext_%j.err

# Finetune ConvNeXt-T stage-4 + head with stage-3 unrolled to R repeats
# (stem/S1/S2/S3 frozen). Default: R=3, Δt=1/3 → T=9 conserved.
#
# Submit from the repo root:
#   source .env && sbatch --account="$SLURM_ACCOUNT" jobs/finetune_s4_r_convnext.sh
#
# Overrides:
#   REPEATS=3 EULER_STEP=0.333333 EPOCHS=50 \
#     sbatch --account="$SLURM_ACCOUNT" jobs/finetune_s4_r_convnext.sh
#   RETAKE=1 sbatch ...   # resume from outputs/<EXPERIMENT_NAME>/weights/last.pth
#
# Broken-horizon control (do NOT use for the main ablation):
#   REPEATS=3 EULER_STEP=1 sbatch ...

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

# ---- experiment knobs (script reads these) ----
export REPEATS="${REPEATS:-3}"
# Default: keep T=9. Leave EULER_STEP unset to let the script use 1/REPEATS.
if [[ -n "${EULER_STEP:-}" ]]; then
  export EULER_STEP
fi
export METHOD="${METHOD:-RK1}"
export WEIGHT_INTERPOLATION="${WEIGHT_INTERPOLATION:-plain}"
export EPOCHS="${EPOCHS:-50}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export LR="${LR:-1e-3}"
export RETAKE="${RETAKE:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-${PROJECT_ROOT}/outputs/convnextv1_imagenet/weights/last.pth}"

# Match scripts/finetune_s4_r_convnext.py naming if not overridden.
_es_for_name="${EULER_STEP:-}"
if [[ -z "${_es_for_name}" ]]; then
  _es_for_name="$(python -c "print(1/${REPEATS})")"
fi
_es_tag="$(python -c "print(f'{float(\"${_es_for_name}\"):g}'.replace('.', 'p'))")"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-convnext_R${REPEATS}_ES${_es_tag}_finetune_s4}"
EXPERIMENT="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"

if [[ "${RETAKE}" == "1" ]]; then
  if [[ ! -f "${EXPERIMENT}/weights/last.pth" ]]; then
    echo "RETAKE=1 but checkpoint not found at ${EXPERIMENT}/weights/last.pth" >&2
    exit 1
  fi
else
  if [[ ! -f "${BASELINE_WEIGHTS}" ]]; then
    echo "Baseline ConvNeXt weights not found at ${BASELINE_WEIGHTS}" >&2
    echo "Train the baseline first, or set BASELINE_WEIGHTS." >&2
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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/finetune_s4_r_convnext.py}"

echo "Host: $(hostname)"
echo "Project: ${PROJECT_ROOT}"
echo "Python: $(which python)"
echo "Venv site: ${VENV_SITE}"
echo "HF cache: ${HF_DATASETS_CACHE}"
echo "Account: ${SLURM_ACCOUNT:-unset}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "REPEATS: ${REPEATS}"
echo "EULER_STEP: ${EULER_STEP:-<script default 1/R>}"
echo "METHOD: ${METHOD}"
echo "WEIGHT_INTERPOLATION: ${WEIGHT_INTERPOLATION}"
echo "EPOCHS: ${EPOCHS} (warmup ${WARMUP_EPOCHS})"
echo "LR: ${LR}"
echo "RETAKE: ${RETAKE}"
echo "Baseline weights: ${BASELINE_WEIGHTS}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "Experiment path: ${EXPERIMENT}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Start: $(date)"
python -c "import torch, torchvision, timm; print(f'torch={torch.__version__} ({torch.__file__})'); print(f'torchvision={torchvision.__version__} ({torchvision.__file__})'); print(f'timm={timm.__version__} ({timm.__file__})')"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS:-}

echo "End: $(date)"
