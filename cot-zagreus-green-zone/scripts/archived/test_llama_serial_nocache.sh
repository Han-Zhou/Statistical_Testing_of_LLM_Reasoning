# Serial + nocache (debug_nocache only, no batch)

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model llama \
    --prompt_type 2 \
    --sample_size 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 3 \
    --temperature 0.9 \
    --tag test-llama-bm-622-serial-nocache \
    --debug_nocache \
    --debug_top20
