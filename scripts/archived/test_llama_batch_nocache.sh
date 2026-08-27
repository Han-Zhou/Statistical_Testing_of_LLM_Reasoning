# Batched + nocache (experimental_llama_batch + debug_nocache)

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model llama \
    --prompt_type 2 \
    --sample_size 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 100 \
    --temperature 0.9 \
    --tag test-llama-bm-622-batch-nocache \
    --experimental_llama_batch \
    --debug_nocache \
    --debug_top20
