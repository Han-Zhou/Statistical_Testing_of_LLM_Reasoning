#!/bin/bash
#SBATCH --job-name=llama_bm_p2
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.err



set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot
scripts/llama/bm/llama_bm_p2.sh
