# NOTE
# tests:
# - generation
# - vanilla confidence
#
# Pickle not generated yet; uncomment --from_pickle once it exists.
# Without it, the dataset loads normally and --sample_size 2 takes the first 2.

python3 main.py \
    --backend api \
    --dataset bigbench_movie \
    --from_pickle /media/data1/han/data_cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model qwen_ascend \
    --sample_size 2 \
    --prompt_type 2 \
    --nb_cot_samples 2 \
    --nb_stepbootstrap_samples 2 \
    --temperature 0.9 \
    --tag new-test-qwen_ascend \
    --debug_top20
