#!/usr/bin/env bash
set -euo pipefail

# Serial Qwen3.6 FP8 smoke test covering generation and confidence paths.
QWEN_FP8_PICKLE_PATH="${QWEN_FP8_PICKLE_PATH:-/shared_work/han/storage/cot/pickles/bigbench_movie_250.pkl}"

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle "${QWEN_FP8_PICKLE_PATH}" \
    --max_tokens 1024 \
    --model qwen_fp8 \
    --sample_size 1 \
    --prompt_type 2 \
    --nb_cot_samples 1 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag test-qwen-fp8-serial \
    --debug_top20
