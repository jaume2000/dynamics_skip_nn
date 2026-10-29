# Matched tied/untied seeds (2 GPUs × 24h, global batch 1024 via accum×2)

# On Leonardo, from the repo root:
source .env

# Three parallel half-node jobs (existing tied D9 counts as the other tied replica):
MODEL=untied SEED=0 sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
MODEL=untied SEED=1 sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
MODEL=tied   SEED=1 sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh

# Outputs:
#   outputs/convnextv1_matched_seed0/
#   outputs/convnextv1_matched_seed1/
#   outputs/shared_convnextv1_matched_seed1/
# (+ existing outputs/shared_convnextv1_imagenet/ as unseeded tied replica)

# Resume after time limit (example):
MODEL=untied SEED=0 RETAKE=1 sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_matched_recipe.sh
squeue -u $USER -o "%.18i %.9P %j %.8u %.2t %.10M %.6D %R"

# Attach to a running job:
# srun --jobid=<ID> --overlap --pty bash

# scratch_full (c≡1) with δ weight decay 0.20 (baseline used 0.05):
VARIANT=scratch_full DELTA_WEIGHT_DECAY=0.20 \
  DELTA_EXPERIMENT_NAME=delta_scratch_full_e300_dwd0p20 \
  sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_delta_experiment.sh
# Output: outputs/delta_scratch_full_e300_dwd0p20/
# Resume: same + RETAKE=1

# 3 shared (W,δ) blocks × 3 repeats each (block-local shared δ), scratch recipe:
#   LR 1e-3, warmup 10, global batch 1024, WD=δWD=0.05, half node (2 GPUs)
VARIANT=scratch_grouped3x3 source .env && \
  sbatch --account="$SLURM_ACCOUNT" --export=ALL jobs/train_delta_experiment.sh
# Output: outputs/delta_scratch_grouped3x3_e300/
# Resume: same + RETAKE=1

cd ~/Projects/ML_and_DL/DeltaNeuralODE
rsync -avz --progress --exclude 'weights/' \
  leonardo:~/convnext_v1/outputs/ \
  outputs/

rsync -avz --progress \
  leonardo:/leonardo_work/AIFAC_P02_1564/feature_maps/featureMaps_interpoled/delta_* \
  outputs/featureMaps_interpoled/
