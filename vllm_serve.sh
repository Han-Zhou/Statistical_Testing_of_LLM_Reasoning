vllm serve Qwen/Qwen3.5-27B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --quantization fp8 \
  --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}'
