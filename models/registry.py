
from types import MappingProxyType
from models.adapters.llama_adapter import LlamaAdapter


MODEL_ADAPTER_REGISTRY = MappingProxyType({
    "llama": LlamaAdapter
})

