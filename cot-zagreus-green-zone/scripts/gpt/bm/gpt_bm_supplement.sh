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
  --nb_cot_samples 32 \
  --nb_stepbootstrap_samples 100 \
  --temperature 0.9 \
  --tag gpt-bm-supplement-623 \
  --discord \
  --debug_top20 \
  --sample_indices /storage/backup/han/backup_workspace/cot-zagreus/gpt_retry_indices_623.txt
