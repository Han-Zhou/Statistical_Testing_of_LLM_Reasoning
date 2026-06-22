#!/bin/bash

srun \
    --job-name=debug_qwen_cache \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --time=2:00:00 \
    --mem=256G \
    --cpus-per-task=24 \
    --exclude=lux-2-node-25 \
    bash -c "
        set -euo pipefail
        module load cuda/12.8
        module load conda
        conda activate cot_vllm
        python3 temp_qwen_cache.py
    "
