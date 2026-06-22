#!/bin/bash

srun \
    --job-name=debug_qwen_vllm \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:2 \
    --time=18:00:00 \
    --mem=256G \
    --cpus-per-task=24 \
    --exclude=lux-2-node-25 \
    bash -c "
        set -euo pipefail
        module load cuda/12.8
        module load conda
        conda activate cot_vllm
        scripts/test_qwen_vllm_generation.sh
    "


    # --nodelist=lux-2-node-18 \


    # --gres=gpu:0 \







