import copy
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.cache_utils import DynamicCache
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
except ImportError:
    # transformers >=5.5.1 dropped Qwen3_5DynamicCache; the qwen3_5 module now
    # uses the generic DynamicCache. The cot_vllm env (transformers 5.10.2) hits
    # this path; the cot env (transformers 5.3.0) still has the dedicated class.
    Qwen3_5DynamicCache = DynamicCache
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from vllm import RequestOutput

from .confidence import ConfidenceTime


from domain.confidence import ConfidenceScores
from domain.evaluation import EvaluationResult


KVCache: TypeAlias = DynamicCache | Qwen3_5DynamicCache

ListAnswerTokenProbs: TypeAlias = list[dict[int, float]]


@dataclass(frozen=True)
class CacheBundle:
    """
    A KV cache bundled with the input_ids it was computed from.
    """

    cache: KVCache
    input_ids: torch.Tensor

    def __post_init__(self):
        if self.input_ids.dim() != 1:
            raise ValueError(
                f"input_ids must be 1D, got shape {tuple(self.input_ids.shape)}"
            )
        cache_len = self.cache.get_seq_length()
        if self.input_ids.shape[0] != cache_len:
            raise ValueError(
                f"input_ids length ({self.input_ids.shape[0]}) must match "
                f"cache.get_seq_length() ({cache_len})"
            )

    def longest_common_prefix(self, new_ids: torch.Tensor) -> int:
        if new_ids.dim() != 1:
            raise ValueError(f"new_ids must be 1D, got shape {tuple(new_ids.shape)}")
        limit = min(new_ids.shape[0], self.input_ids.shape[0])
        if limit == 0:
            return 0
        cached = self.input_ids[:limit].to(new_ids.device)
        mismatches = new_ids[:limit].ne(cached).nonzero(as_tuple=False)
        return limit if mismatches.numel() == 0 else int(mismatches[0].item())


@dataclass
class AnswerSpan:
    char_answer_sentence_start: int
    char_answer_boxed_start: int
    char_answer_boxed_end: int


@dataclass
class Datapoint:
    id: str
    question: str
    ground_truth: str
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: defaultdict(list))


@dataclass
class PromptRequest:
    # few-shot not implemented yet
    few_shot: bool
    prompt_type: int


@dataclass
class Timings:
    """Timings for generation and confidence, used for each sampling method:
    - vanilla
    - rejection
    - lawyer
    - stepbootstrap
    """

    generation_time: float
    confidence_time: ConfidenceTime


@dataclass
class TrajectoryRecord:
    id: str
    question: str
    prompt: str
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    ground_truth: str
    evaluation_result: EvaluationResult | None = None
    prompt_cache_path: Path | None = None
    confidences: ConfidenceScores | None = None
    timings: Timings | None = None
    input_messages: list[dict[str, str]] | None = None
    cost: float = 0.0
    

@dataclass
class PregeneratedTrajectoryRecord:
    id: str
    prompt: str
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    prompt_cache_path: Path | None = None


@dataclass
class ParsedOutputGeneration:
    cot_steps: list[str]
    final_answer: str
    text_question: str
    text_cot: str
    text_cot_with_answer: str
    whole_cache: CacheBundle
    question_cache: CacheBundle
    answer_token_probs: torch.Tensor | list[list[TopLogprob]]
    answer_token_ids: torch.Tensor | list[str]

    # experimental - stats for scores
    answer_token_score_probs: torch.Tensor | None = None

    # API path only: original chat-completions messages used to produce this output
    input_messages: list[dict[str, str]] | None = None




@dataclass
class LLMOutput:
    outputs: GenerateDecoderOnlyOutput | ChatCompletion | RequestOutput
    offset_mappings: list[tuple[int, int]] | None
    text_question: str | None = None
    input_messages: list[dict[str, str]] | None = None



