from dataclasses import dataclass



@dataclass
class ConfidenceScores:
    """All confidence scores computed for a single generation."""
    answer_probabilities: list[dict[str, float]]
    answer_entropy: list[dict[str, float]]
    indirect_probabilities: list[dict[str, float]]
    verbconf_probabilities: list[float]
    verbconf_distribution: list[float] | None = None   # softmaxed probs for scores 0-100 (length 101)
    verbconf_top_score: int | None = None               # score with highest probability
    verbconf_top_prob: float | None = None              # probability of that score
    step_masks: list[list[int]] | None = None           # binary list per sample: 1 = kept, 0 = masked


