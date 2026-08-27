#!/bin/bash
#SBATCH --job-name=llama_bm_p2
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=.slurm_logs/%j_%x.out
#SBATCH --error=.slurm_logs/%j_%x.err



set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/discord_notify.sh"
module load cuda/12.4
module load conda
conda activate cot
scripts/llama/bm/llama_bm_p2.sh
notify_discord "[SUCCESS] ${discord_job_name} completed successfully on $(hostname) (job ${SLURM_JOB_ID:-local})."
