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
