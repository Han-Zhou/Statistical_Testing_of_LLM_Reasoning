#!/bin/bash

srun \
    --job-name=qwen_flash \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:2 \
    --time=2:00:00 \
    --mem=128G \
    --cpus-per-task=24 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        python3 temp_qwen_flash.py
    "
