
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Optional, Tuple

from models.core_models import LLM, API_LLM
from domain.data import LLMOutput, ParsedOutputGeneration, KVCache


Mode = Literal["generation", "confidence"]


@dataclass
class AnswerSpan:
    char_answer_sentence_start: int
    char_answer_boxed_start: int
    char_answer_boxed_end: int



class ModelAdapter(ABC):
    def __init__(self):
        self.model: LLM | API_LLM

    @abstractmethod
    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        ...
    
    @abstractmethod
    def process_generation_output(self, llm_outputs: LLMOutput) -> ParsedOutputGeneration:
        ...
    
    # @abstractmethod
    # def parse_output_answer_suffix(self, llm_outputs):
    #     ...
    
    def generate(
            self, 
            messages: list[dict[str, str]],
            max_tokens: int, 
            cache: Optional[tuple] = None, 
            # stop_strings: list[str] = None, 
            temperature: float = 0.0
        ) -> ParsedOutputGeneration:
        prompt_text = self.render_prompt(messages)
        output = self.model.generate(
            prompt=prompt_text,
            max_tokens=max_tokens,
            cache=cache,
            # stop_strings=stop_strings,
            temperature=temperature
        )
        return self.parse_generation_output(output)



