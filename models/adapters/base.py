
from abc import ABC, abstractmethod
from typing import Literal, Protocol, Sequence, Optional, Tuple

from models.core_models import LLM, API_LLM
from domain import LLMOutput, ParsedOutputGeneration, CacheBundle, ScorerOutput


Mode = Literal["generation", "confidence"]


"""
ModelScorer handles the confidence scoring - Indirect and Verbal
"""
class ModelScorer(ABC):
    def __init__(self, model: LLM | API_LLM):
        self.model: LLM | API_LLM = model

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
        self.model: LLM | API_LLM
        self.model_scorer: ModelScorer

    @abstractmethod
    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> Optional[CacheBundle]:
        ...


    @abstractmethod
    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        ...
    
    @abstractmethod
    def process_generation_output(self, llm_outputs: LLMOutput) -> ParsedOutputGeneration:
        ...
    
    
    def generate(
            self, 
            messages: list[dict[str, str]],
            max_tokens: int, 
            cache: Optional[CacheBundle] = None,
            # stop_strings: list[str] = None, 
            temperature: float = 0.0
        ) -> ParsedOutputGeneration:
        prompt_text = self.render_prompt(messages)
        cache = self.align_cache(cache, prompt_text)
        output = self.model.generate(
            prompt=prompt_text,
            max_tokens=max_tokens,
            cache=cache,
            # stop_strings=stop_strings,
            temperature=temperature
        )
        return self.process_generation_output(output)


    def scorer(self) -> ModelScorer:
        return self.model_scorer




