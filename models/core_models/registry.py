
from types import MappingProxyType



MODEL_HF_REGISTRY = MappingProxyType({
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3.5-27B",
    # "mistral": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
})


MODEL_API_REGISTRY = MappingProxyType({
    "gpt": "gpt-4o-2024-11-20"
})




