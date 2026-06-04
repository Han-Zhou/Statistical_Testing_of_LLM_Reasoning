# NOTE
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend hf \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model gpt \
    --prompt_type 2 \
    --nb_cot_samples 4 \
    --nb_stepbootstrap_samples 4 \
    --sample_size 2 \
    --temperature 0.9 \
    --tag test-gpt-601  \
    # --discord

