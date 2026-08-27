#!/bin/bash
# Runs the end-to-end cache-bug demo (temp_demo_cache_bug_e2e.py) through the
# real LlamaAdapter / LLM.align_cache / LlamaScorer. Loads Llama-3.1-8B -> GPU.

srun \
    --job-name=demo_cache_bug_e2e \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:1 \
    --time=1:00:00 \
    --mem=128G \
    --cpus-per-task=24 \
    --exclude=lux-2-node-25 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        python3 temp_demo_cache_bug_e2e.py
    "
