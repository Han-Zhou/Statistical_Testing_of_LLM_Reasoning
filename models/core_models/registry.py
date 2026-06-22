
from types import MappingProxyType



MODEL_HF_REGISTRY = MappingProxyType({
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3.5-27B",
    "qwen_vllm": "Qwen/Qwen3.5-27B",
    # "mistral": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
})


MODEL_API_REGISTRY = MappingProxyType({
    # "gpt": "gpt-4o-2024-11-20"
    # "gpt": "gpt-4.1-mini"
    "gpt": "gpt-4o-mini-2024-07-18"
})


MODEL_API_PRICING = MappingProxyType({
    # "gpt-4o-2024-11-20": {"input": 5.0 / 1e6, "output": 20.0 / 1e6},
    # "gpt-4.1-mini": {"input": 0.8 / 1e6, "output": 3.2 / 1e6},
    "gpt-4o-mini-2024-07-18": {"input": 0.3 / 1e6, "output": 1.2 / 1e6},
})




