import logging
import copy
import torch
from typing import Optional, Tuple
from dotenv import load_dotenv

from vllm import LLM, SamplingParams
from vllm import RequestOutput

from transformers import AutoTokenizer
# from transformers.utils import ModelOutput
# from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
# from transformers.cache_utils import DynamicCaache

from domain.data import LLMOutput, KVCache, CacheBundle
from models.core_models.registry import MODEL_HF_REGISTRY

load_dotenv()
logger = logging.getLogger(__name__)

class VLLM_LLM():

    def __init__(self, model_name: str):
        self.model_name = model_name
        # self.attention_implementation = attention_implementation
        self._load_model()


    def _load_model(self):
        # Load the model based on the model name
        logger.info(f"Loading VLLM_LLM model: {self.model_name}")
        actual_model_name = MODEL_HF_REGISTRY.get(self.model_name)
        if not actual_model_name:
            raise ValueError(f"Model {self.model_name} not found in registry.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name)

        hf_overrides =None
        if self.model_name == "qwen":
            hf_overrides = {"architectures": ["Qwen3_5ForCausalLM"]}

        self.model = LLM(
            model=actual_model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.bfloat16,
            quantization="fp8",
            hf_overrides=hf_overrides,
        )
        
        logger.info(f"Model {self.model_name} loaded successfully.")


    def generate(
            self, 
            prompts: list[str], 
            max_tokens: int, 
            temperature: float,
            n: int,
            stop_strings: list[str] | None = None,
            include_stop_str_in_output: bool = False,
        ) -> list[LLMOutput]:
        # Generate text based on the prompt
        # prompt SHOULD ALREADY HAVE chat template applied
        logger.info(f"Generating text with model {self.model_name}")

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strings,
            include_stop_str_in_output=include_stop_str_in_output,
            logprobs=20,
            skip_special_tokens=False,
            n=n,
        )

        outputs: list[RequestOutput] = self.model.generate(prompts, sampling)
        
        return [LLMOutput(outputs=output, offset_mappings=None) for output in outputs]
    


    def forward(
        self,
        prompts: list[str],
        return_llm_output: bool = False,
    ) -> list[RequestOutput] | list[LLMOutput]:
        """
        Single forward pass.
        - return_llm_output=False: returns the raw RequestOutput (used for confidence scoring).
        - return_llm_output=True: returns the LLMOutput
        """
        logger.info(f"Forward pass with model {self.model_name}")

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=0,
            prompt_logprobs=20,
            skip_special_tokens=False,
        )

        outputs: list[RequestOutput] = self.model.generate(prompts, sampling)

        if not return_llm_output:
            return outputs

        return [LLMOutput(outputs=output, offset_mappings=None) for output in outputs]


    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> None:
        """
        we don't support caching for vllm
        """
        return None