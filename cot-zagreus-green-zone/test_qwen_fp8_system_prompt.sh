# conda activate cot_vllm

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc-per-node=2 \
  test_qwen_fp8_system_prompt.py \
  --sample_size 1

  