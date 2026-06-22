#!/bin/bash
#SBATCH --job-name=gpt_bm
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --nodelist=lux-2-node-18
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
# SBATCH --exclude=lux-2-node-21,lux-2-node-23

#SBATCH --output=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.err



set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot
scripts/gpt/bm/gpt_bm.sh