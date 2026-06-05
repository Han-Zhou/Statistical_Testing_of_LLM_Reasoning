# NOTE
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend api \
    --dataset bigbench_movie \
    --from_pickle /storage/backup/han/cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model gpt \
    --prompt_type 2 \
    --nb_cot_samples 4 \
    --nb_stepbootstrap_samples 4 \
    --sample_size 5 \
    --temperature 0.9 \
    --tag test-gptmini-605  \
    --debug_top20
    # --discord

