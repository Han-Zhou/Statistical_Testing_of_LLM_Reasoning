from dataclasses import dataclass
from typing import Any

import torch


# Convention: `debug` fields below are free-form dicts populated only when a
# debug flag on ConfidenceConfig is on. Add new measurements by writing a key
# into the local dict in ConfidenceEngine; promote a key to a typed field once
# its shape stabilizes. JSON serialization is automatic via asdict.
@dataclass
class ConfidenceTime:
    answer_prob_time: float
    answer_ent_time: float
    indirect_time: float
    verbconf_time: float
    # experimental - stats for scores
    answer_score_prob_time: float | None = None
    answer_score_ent_time: float | None = None
    debug: dict[str, Any] | None = None


@dataclass
class ConfidenceScores:
    """All confidence scores computed for a single generation."""
    answer_probabilities: list[dict[str, float]]
    answer_entropy: list[dict[str, float]]
    indirect_probabilities: list[dict[str, float]]
    verbconf_probabilities: list[float]
    # verbconf_distribution: list[float] | None = None   # softmaxed probs for scores 0-100 (length 101)
    # verbconf_top_score: int | None = None               # score with highest probability
    # verbconf_top_prob: float | None = None              # probability of that score
    step_masks: list[list[int]] | None = None           # binary list per sample: 1 = kept, 0 = masked

    # experimental - stats for scores
    answer_score_probabilities: list[dict[str, float]] | None = None
    answer_score_entropy: list[dict[str, float]] | None = None

    debug: dict[str, Any] | None = None


ScorerOutput = dict[str, torch.Tensor]







