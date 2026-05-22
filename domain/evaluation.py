from dataclasses import dataclass

@dataclass
class EvaluationResult:
    correct: bool
    score: float | None = None
    llm_judge_applicable: bool | None = None
    