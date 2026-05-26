#!/bin/bash

srun \
    --job-name=test_logits_shape \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:1 \
    --time=0:30:00 \
    --mem=64G \
    --cpus-per-task=8 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        python3 test_logits_shape_mismatch.py --model llama
    "
