#!/bin/bash

srun \
    --job-name=debug_vllm_stop_tok \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --time=2:00:00 \
    --gres=gpu:2 \
    --mem=256G \
    --cpus-per-task=24 \
    bash -c "
        set -euo pipefail
        module load cuda/12.8
        module load conda
        conda activate cot_vllm
        python3 temp_vllm_stop_tok.py
    "
