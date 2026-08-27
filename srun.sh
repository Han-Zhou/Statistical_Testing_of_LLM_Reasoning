#!/bin/bash

srun \
    --job-name=debug_llama \
    --qos=high \
    --partition=compute \
    --nodes=1 \
    --gres=gpu:1 \
    --time=18:00:00 \
    --mem=128G \
    --cpus-per-task=24 \
    bash -c "
        set -euo pipefail
        module load cuda/12.4
        module load conda
        conda activate cot
        scripts/test_llama_generation.sh
    "


    # --nodelist=lux-2-node-18 \

    # --exclude=lux-2-node-25 \

    # --gres=gpu:0 \

