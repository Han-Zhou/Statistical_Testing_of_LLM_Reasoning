from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from domain.confidence import ConfidenceScores



@dataclass
class Datapoint:
    id: str
    question: str
    ground_truth: str
    context: str | None = None
    metadata: dict[str, Any] = defaultdict


@dataclass
class EvaluationResult:
    correct: bool
    score: float | None = None
    llm_judge_applicable: bool | None = None


@dataclass
class PromptRequest:
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


@dataclass
class PregeneratedTrajectoryRecord:
    id: str
    prompt: str
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    prompt_cache_path: Path | None = None


