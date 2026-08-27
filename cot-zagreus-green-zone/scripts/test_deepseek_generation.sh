# Smoke test for the deepseek-v4-flash API model.
#
# Run from the repo root:
#   OPENAI_BASE_URL=<gateway> OPENAI_API_KEY=<key> bash scripts/test_deepseek_generation.sh
#
# Set OPENAI_BASE_URL / OPENAI_API_KEY to the Deepseek-serving gateway in your
# .env (or export inline). Pricing is $0.0 in the registry until real rates are
# known.

python3 main.py \
    --backend api \
    --dataset bigbench_movie \
    --from_pickle /media/data1/han/data_cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model deepseek_flash \
    --sample_size 2 \
    --prompt_type 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag test-deepseek_flash \
    --debug_top20
