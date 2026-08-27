# NOTE
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend vllm \
    --dataset bigbench_movie \
    --from_pickle /shared_work/han/storage/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model qwen_vllm \
    --sample_size 8 \
    --prompt_type 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag test-810-qwen_vllm \
    --experimental_batch \
    --vllm_base_url http://127.0.0.1:8000 \
    --debug_top20

