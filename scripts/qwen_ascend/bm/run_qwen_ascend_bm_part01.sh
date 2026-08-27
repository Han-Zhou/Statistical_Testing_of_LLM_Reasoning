#!/bin/bash
#SBATCH --job-name=qwen_ascend_bm_p01
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --exclude=lux-2-node-25

#SBATCH --output=.slurm_logs/%j_%x.out
#SBATCH --error=.slurm_logs/%j_%x.err



set -euo pipefail
# source "${SLURM_SUBMIT_DIR:-.}/scripts/discord_notify.sh"
# module load cuda/12.4
# module load conda
source /home/han/miniconda3/etc/profile.d/conda.sh
conda activate cot-zagreus-batching
scripts/qwen_ascend/bm/qwen_ascend_bm_part01.sh
# notify_discord "[SUCCESS] ${discord_job_name} completed successfully on $(hostname) (job ${SLURM_JOB_ID:-local})."
