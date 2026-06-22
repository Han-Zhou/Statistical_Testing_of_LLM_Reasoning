#!/bin/bash
#SBATCH --job-name=test_llama_nocache
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=7-8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.err



set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot
# scripts/test_llama_generation.sh
scripts/tl2.sh
# scripts/test_qwen_generation.sh
# scripts/test_gpt_generation.sh


#   SBATCH --gres=gpu:1




