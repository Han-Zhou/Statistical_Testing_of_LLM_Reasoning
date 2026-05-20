
from types import MappingProxyType


MODEL_ATTENTION_IMPLEMENTATION_REGISTRY = MappingProxyType({
    "llama": "sdpa"
})
