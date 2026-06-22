
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



    def scorer(self) -> ModelScorer:
        return self.model_scorer




