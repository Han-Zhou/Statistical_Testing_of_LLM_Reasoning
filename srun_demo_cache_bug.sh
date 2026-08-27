#!/bin/bash
# Runs the no-GPU tokenizer-only cache-bug demo (temp_demo_cache_bug.py).
# Needs the `cot` env for the Llama tokenizer; no weights/GPU loaded.
# Optionally pass a trajectory JSON: bash srun_demo_cache_bug.sh path/to/traj.json

srun \
    --job-name=demo_cache_bug \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:0 \
    --time=0:30:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --exclude=lux-2-node-25 \
    bash -c "
        set -euo pipefail
        module load conda
        conda activate cot
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        python3 temp_demo_cache_bug.py $*
    "
