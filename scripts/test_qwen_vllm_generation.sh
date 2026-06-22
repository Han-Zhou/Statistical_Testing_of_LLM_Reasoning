# NOTE
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend vllm \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model qwen_vllm \
    --sample_size 2 \
    --prompt_type 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag test-615-qwen_vllm \
    --debug_top20


