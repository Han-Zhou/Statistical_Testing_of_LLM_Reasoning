#!/bin/bash
#SBATCH --job-name=llama-bm-batch
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --output=.slurm_logs/%j_%x.out
#SBATCH --error=.slurm_logs/%j_%x.err

set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot

bash scripts/llama/bm/llama_bm_batch.sh
