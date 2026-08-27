#!/bin/bash
# Interactive srun for the stepbootstrap confidence cache-equivalence test.
# Runs temp_test_sb_conf.py on one GPU; output streams to this terminal.

srun \
    --job-name=debug_sb_conf \
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
        python3 temp_test_sb_conf.py
    "
