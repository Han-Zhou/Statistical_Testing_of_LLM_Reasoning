import copy
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from transformers.cache_utils import DynamicCache


from domain.confidence import ConfidenceScores
from domain.evaluation import EvaluationResult



KVCache: TypeAlias = DynamicCache | Qwen3_5DynamicCache


@dataclass(frozen=True)
class CachedPrefix:
    """
    A KV cache bundled with the input_ids it was computed from.
    """
    cache: KVCache
    input_ids: torch.Tensor

    def __post_init__(self):
        if self.input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1D, got shape {tuple(self.input_ids.shape)}")
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
    confidence_time: float


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


@dataclass
class PregeneratedTrajectoryRecord:
    id: str
    prompt: str
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    prompt_cache_path: Path | None = None


# NOTE finish this dataclass
@dataclass
class ParsedOutputGeneration:
    cot_steps: list[str]
    final_answer: str
    text_question: str
    text_cot: str
    text_cot_with_answer: str
    cot_with_answer_cache: KVCache
    question_prefix: CachedPrefix
    answer_token_probs: torch.Tensor



@dataclass
class LLMOutput:
    outputs: GenerateDecoderOnlyOutput
    offset_mappings: list[tuple[int, int]] | None



