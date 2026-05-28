#!/bin/bash
#SBATCH --job-name=debug_qwen
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=18:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24

set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot
scripts/test_qwen_generation.sh
