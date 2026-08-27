#!/bin/bash
#SBATCH --job-name=test_qwen_vllm
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.err



set -euo pipefail
module load cuda/12.8
module load conda
conda activate cot_vllm
scripts/test_qwen_vllm_generation.sh



#   SBATCH --gres=gpu:1




