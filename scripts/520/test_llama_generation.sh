python3 -m main \
    --backend hf \
    --confidence none \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie.pkl \
    --max_tokens 512 \
    --model llama \
    --sample_size 2 \
    --prompt_type 2
