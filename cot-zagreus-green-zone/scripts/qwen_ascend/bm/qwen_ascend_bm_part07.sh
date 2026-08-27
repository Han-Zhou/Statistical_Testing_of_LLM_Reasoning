# NOTE
# bm part 07: datapoints 150-175
# tests:
# - generation
# - vanilla confidence

python3 main.py \
    --backend api \
    --dataset bigbench_movie \
    --from_pickle /media/data1/han/data_cot/pickles/bigbench_movie_250.pkl \
    --max_tokens 1024 \
    --model qwen_ascend \
    --prompt_type 2 \
    --nb_cot_samples 32 \
    --nb_stepbootstrap_samples 100 \
    --temperature 0.9 \
    --tag qwen-ascend-bm-825-part07  \
    --sample_range 150 175 \
    --debug_top20
        # --discord \
