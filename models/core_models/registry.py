
from types import MappingProxyType



MODEL_HF_REGISTRY = MappingProxyType({
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3.5-27B",
    # "qwen_fp8": "Qwen/Qwen3.6-27B-FP8",
    "qwen_fp8": "Qwen/Qwen3.8-27B-FP8",
    "qwen_vllm": "Qwen/Qwen3.5-27B",
    # "mistral": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
})


MODEL_API_REGISTRY = MappingProxyType({
    # "gpt": "gpt-4o-2024-11-20"
    # "gpt": "gpt-4.1-mini"
    "gpt": "gpt-4o-mini-2024-07-18",
    "qwen_ascend": "Qwen3.8-27B",
    "deepseek_flash": "deepseek-v4-flash-0731",
})


MODEL_API_PRICING = MappingProxyType({
    # "gpt-4o-2024-11-20": {"input": 5.0 / 1e6, "output": 20.0 / 1e6},
    # "gpt-4.1-mini": {"input": 0.8 / 1e6, "output": 3.2 / 1e6},
    "gpt-4o-mini-2024-07-18": {"input": 0.3 / 1e6, "output": 1.2 / 1e6},
    "gpt-4o": {"input": 5.0 / 1e6, "output": 20.0 / 1e6},
    "Qwen3.8-27B": {"input": 0.0, "output": 0.0},
    "deepseek-v4-flash-0731": {"input": 0.0, "output": 0.0},
})

