#!/bin/bash
#SBATCH --job-name=qwen_vllm
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24

#SBATCH --output=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.out
#SBATCH --error=/storage/backup/han/backup_workspace/cot-zagreus/.slurm_logs/%j_%x.err



set -euo pipefail
module load cuda/12.8
module load conda
conda activate cot_vllm

export MAX_JOBS=8
export NVCC_THREADS=1

python3 temp_qwen_vllm.py
