from types import MappingProxyType
from models.adapters.llama_adapter import LlamaAdapter
from models.adapters.qwen_adapter import QwenAdapter, QwenFp8Adapter
from models.adapters.gpt_adapter import GptAdapter
from models.adapters.qwen_vllm_adapter import QwenVllmAdapter
from models.adapters.qwen_ascend_adapter import QwenAscendAdapter
from models.adapters.deepseek_adapter import DeepseekAdapter

MODEL_ADAPTER_REGISTRY = MappingProxyType(
    {
        "llama": LlamaAdapter,
        "qwen": QwenAdapter,
        "qwen_fp8": QwenFp8Adapter,
        "qwen_vllm": QwenVllmAdapter,
        "gpt": GptAdapter,
        "qwen_ascend": QwenAscendAdapter,
        "deepseek_flash": DeepseekAdapter,
    }
)
