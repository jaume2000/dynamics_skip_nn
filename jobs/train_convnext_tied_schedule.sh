#!/bin/bash
# Untied ConvNeXt-T with the weight-tied recipe (scripts/train_convnextV1_tied_schedule.py),
# 4 GPUs x 256, chained 12 h jobs exactly like the tied run (jobs/train_delta_convnext.sh).
#
#   source .env && bash jobs/train_convnext_tied_schedule.sh            # first 12 h chunk
#   later chunks: set RETAKE=1 and EXPERIMENT_NAME=convnextv1_imagenet_tied_schedule in .env
#   (the parent job sources .env and checks outputs/$EXPERIMENT_NAME/weights/last.pth)

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAIN_SCRIPT="scripts/train_convnextV1_tied_schedule.py"
export NPROC_PER_NODE=4
exec bash "${PROJECT_ROOT}/jobs/train_delta_convnext.sh" "$@"
