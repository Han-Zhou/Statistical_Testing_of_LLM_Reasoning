

from abc import ABC, abstractmethod

from typing import Any

import torch

from domain import ParsedOutputGeneration, ConfidenceScores, ScorerOutput
from models.adapters import ModelScorer

    


class ConfidenceMethod(ABC):
    def __init__(self, model_scorer: ModelScorer):
        self.model_scorer = model_scorer

    @abstractmethod
    def tail_prompt(self, final_answer: str | int) -> str:
        """Unique suffix appended after the prompt for this confidence method"""
        ...

    @abstractmethod
    def extract(self, scorer_output: ScorerOutput) -> dict[str, float]:
        """extract and process the logits we are interested in for this confidence method"""
        ...

    @abstractmethod
    def compute_confidence(
        self,
        parsed_output: ParsedOutputGeneration,
    ) -> dict[str, float]:
        ...

