
from abc import ABC, abstractmethod
from typing import Literal, Protocol, Sequence, Optional, Tuple

from transformers.utils import ModelOutput

from models.core_models import LLM, API_LLM, VLLM_LLM

from domain import LLMOutput, ParsedOutputGeneration, CacheBundle, ScorerOutput, KVCache


Mode = Literal["generation", "confidence"]


"""
ModelScorer handles the confidence scoring - Indirect and Verbal
"""
class ModelScorer(ABC):
    def __init__(self, model: LLM | API_LLM | VLLM_LLM):
        self.model: LLM | API_LLM | VLLM_LLM = model

    @abstractmethod
    def forward_indirect(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput:
        ...

    @abstractmethod
    def forward_verbal(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput:
        ...

    def forward_batch_confidence(
        self,
        prompts: list[str],
        shared_cache: Optional[CacheBundle] = None,
    ) -> list:
        """Batched forward pass returning last-token logits for each prompt.
        Override in subclass for batch support. Default raises."""
        raise NotImplementedError("Batched confidence scoring not supported for this model")


"""
ModelAdapter is the adapter between the runner and the core LLM / API_LLM. 
- handles prompt processing and output parsing
- handles generation
- delegates confidence scoring to ModelScorer
"""
class ModelAdapter(ABC):
    def __init__(self):
        self.model: LLM | API_LLM | VLLM_LLM
        self.model_scorer: ModelScorer

    
    def cost(self) -> float:
        return 0.0

    
    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> Optional[KVCache]:
         return self.model.align_cache(cache, prompt_text)


    @abstractmethod
    def render_prompt(self, messages: list[dict[str, str]]) -> str | list[dict[str, str]]:
        ...
    
    @abstractmethod
    def process_generation_output(self, llm_outputs: LLMOutput) -> ParsedOutputGeneration:
        ...

    
    @abstractmethod
    def generate_helper(
        self, 
            prompt: str | list[dict[str, str]], 
            max_tokens: int, 
            cache: Optional[Tuple], 
            temperature: float
        ) -> LLMOutput:
        ...
    

    @abstractmethod
    def forward_pass_helper(
        self,
        prompt: str | list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> LLMOutput | ModelOutput:
        ...

    
    def generate(
        self, 
        messages: list[dict[str, str]],
        max_tokens: int, 
        cache: Optional[CacheBundle] = None,
        temperature: float = 0.0
    ) -> ParsedOutputGeneration | list[ParsedOutputGeneration]:
        prompt_text = self.render_prompt(messages)
        cache = self.align_cache(cache, prompt_text)
        output = self.generate_helper(prompt_text, max_tokens, cache, temperature)
        return self.process_generation_output(output)



    def forward_pass(
        self,
        messages: list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
    ) -> ParsedOutputGeneration | list[ParsedOutputGeneration]:
        prompt_text = self.render_prompt(messages)
        cache = self.align_cache(cache, prompt_text)
        output = self.forward_pass_helper(prompt_text, cache, return_llm_output=True)
        return self.process_generation_output(output)


    def generate_batch(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        cache: Optional[CacheBundle] = None,
        temperature: float = 0.0,
        num_sequences: int = 1,
    ) -> list[ParsedOutputGeneration]:
        """Batched generation: N sequences from the same prompt.
        Requires generate_batch_helper to be implemented by the adapter."""
        prompt_text = self.render_prompt(messages)
        aligned_cache = self.align_cache(cache, prompt_text)
        outputs = self.generate_batch_helper(prompt_text, max_tokens, aligned_cache, temperature, num_sequences)
        return [self.process_generation_output(o) for o in outputs]


    def forward_pass_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        cache: Optional[CacheBundle] = None,
    ) -> list[ParsedOutputGeneration]:
        """Batched forward pass: N different prompts.
        If cache (CacheBundle) is provided and aligns, uses cache-replicated batching."""
        prompts = [self.render_prompt(m) for m in messages_list]
        if cache is not None:
            aligned_cache = self.model.align_cache(cache, prompts[0])
        else:
            aligned_cache = None
        if aligned_cache is not None:
            cache_seq_len = aligned_cache.get_seq_length()
            outputs = self.forward_pass_batch_helper(prompts, cache=aligned_cache, cache_seq_len=cache_seq_len)
        else:
            outputs = self.forward_pass_batch_helper(prompts)
        return [self.process_generation_output(o) for o in outputs]


    def generate_batch_helper(
        self,
        prompt: str,
        max_tokens: int,
        cache: Optional[KVCache],
        temperature: float,
        num_sequences: int,
    ) -> list[LLMOutput]:
        raise NotImplementedError("Batched generation not supported for this model")

    def forward_pass_batch_helper(
        self,
        prompts: list[str],
        cache: Optional[KVCache] = None,
        cache_seq_len: int = 0,
    ) -> list[LLMOutput]:
        raise NotImplementedError("Batched forward pass not supported for this model")



    def scorer(self) -> ModelScorer:
        return self.model_scorer




