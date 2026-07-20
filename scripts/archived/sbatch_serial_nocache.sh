#!/bin/bash
#SBATCH --job-name=llama-serial-nocache
#SBATCH --qos=high
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=24
#SBATCH --output=.slurm_logs/serial_nocache_%j.out
#SBATCH --error=.slurm_logs/serial_nocache_%j.err

set -euo pipefail
module load cuda/12.4
module load conda
conda activate cot

bash scripts/test_llama_serial_nocache.sh
