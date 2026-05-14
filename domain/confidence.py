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


@dataclass
class ConfidenceTimings:
    """Granular timing for confidence computation components."""
    # Vanilla per-metric timing (includes forward pass + post-processing)
    vanilla_meanprob_total_seconds: float = 0.0
    vanilla_meanprob_forward_seconds: float = 0.0

    vanilla_meanent_total_seconds: float = 0.0
    vanilla_meanent_forward_seconds: float = 0.0

    vanilla_indirect1_total_seconds: float = 0.0
    vanilla_indirect1_forward_seconds: float = 0.0

    vanilla_verbconf_total_seconds: float = 0.0
    vanilla_verbconf_forward_seconds: float = 0.0
    vanilla_verbconf_joint_probs_seconds: float = 0.0

    # Bootstrap-specific timing (when experimental_bootstrap=True)
    bootstrap_meanprob_total_seconds: float = 0.0
    bootstrap_meanprob_forward_seconds: float = 0.0

    bootstrap_indirect1_total_seconds: float = 0.0
    bootstrap_indirect1_forward_seconds: float = 0.0

    bootstrap_verbconf_total_seconds: float = 0.0
    bootstrap_verbconf_forward_seconds: float = 0.0
    bootstrap_verbconf_joint_probs_seconds: float = 0.0