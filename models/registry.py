from types import MappingProxyType
from models.adapters.llama_adapter import LlamaAdapter
from models.adapters.qwen_adapter import QwenAdapter
from models.adapters.gpt_adapter import GptAdapter
# from models.adapters.qwen_vllm_adapter import QwenVllmAdapter

MODEL_ADAPTER_REGISTRY = MappingProxyType(
    {
        "llama": LlamaAdapter,
        "qwen": QwenAdapter,
        # "qwen_vllm": QwenVllmAdapter,
        "gpt": GptAdapter,
    }
)
