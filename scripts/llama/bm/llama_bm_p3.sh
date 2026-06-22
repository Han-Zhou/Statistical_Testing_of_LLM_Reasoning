# NOTE
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model llama \
    --prompt_type 2 \
    --nb_cot_samples 32 \
    --nb_stepbootstrap_samples 100 \
    --temperature 0.9 \
    --sample_range 150 200 \
    --tag llama-bm-610-p3  \
    --discord \
    --debug_top20
