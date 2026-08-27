# NOTE
# tests Qwen HF batched:
# - rejection and lawyer generation
# - step-bootstrap forward passes
# - confidence scoring

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle /shared_work/han/storage/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model qwen \
    --sample_size 8 \
    --prompt_type 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag test-810-qwen-hf-batch \
    --experimental_batch \
    --debug_top20
