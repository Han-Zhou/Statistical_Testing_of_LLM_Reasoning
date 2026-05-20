from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from transformers.cache_utils import DynamicCache


from domain.confidence import ConfidenceScores, ConfidenceTimings



KVCache: TypeAlias = DynamicCache | Qwen3_5DynamicCache


@dataclass
class Datapoint:
    id: str
    question: str
    ground_truth: str
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: defaultdict(list))


@dataclass
class EvaluationResult:
    correct: bool
    score: float | None = None
    llm_judge_applicable: bool | None = None


@dataclass
class PromptRequest:
    # few-shot not implemented yet
    few_shot: bool
    prompt_type: int


@dataclass
class TrajectoryRecord:
    id: str
    question: str
    prompt: str
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    ground_truth: str
    correct: bool | None
    prompt_cache_path: Path | None = None
    confidences: ConfidenceScores | None = None
    confidence_timings: ConfidenceTimings | None = None


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
    question_cache: KVCache
    answer_token_probs: torch.Tensor



@dataclass
class LLMOutput:
    outputs: GenerateDecoderOnlyOutput
    offset_mappings: list[tuple[int, int]] | None



