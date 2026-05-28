
from types import MappingProxyType


MODEL_ATTENTION_IMPLEMENTATION_REGISTRY = MappingProxyType({
    # "llama": "sdpa"
    "llama": "flash_attention_2",
    "qwen": "flash_attention_2",
})

ANSWER_TOKENS = {
    # ' Yes': [' Yes', ' yes', ' YES', ' Yeah', ' yeah', ' Yep', ' yep'],
    # ' No': [' No', ' no', ' NO', ' Nah', ' nah', ' Nope', ' nope'],
    ' True': [' True'],
    ' False': [' False'],
    'llama_verbal_confidence': [str(i) for i in range(0, 101)]
}
