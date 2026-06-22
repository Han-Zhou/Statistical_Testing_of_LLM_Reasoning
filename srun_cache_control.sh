#!/bin/bash
# Interactive srun for the KV-cache negative-control test.
# Proves the cache is read (not a silent no-op) by corrupting its contents.

srun \
    --job-name=debug_cache_ctrl \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:1 \
    --time=1:00:00 \
    --mem=128G \
    --cpus-per-task=24 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        python3 temp_test_cache_control.py
    "
