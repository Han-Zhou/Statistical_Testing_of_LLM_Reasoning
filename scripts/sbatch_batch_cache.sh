#!/bin/bash
#SBATCH --job-name=llama-batch-cache
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --output=.slurm_logs/batch_cache_%j.out
#SBATCH --error=.slurm_logs/batch_cache_%j.err

set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot

bash scripts/test_llama_batch_cache.sh
