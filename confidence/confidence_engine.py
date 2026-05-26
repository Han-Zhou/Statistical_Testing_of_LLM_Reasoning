
import torch

from config import ConfidenceConfig
from domain import ParsedOutputGeneration, ConfidenceScores
from confidence.indirect import IndirectConfidenceMethod
from confidence.verbal import VerbalConfidenceMethod
from models.adapters import ModelScorer

class ConfidenceEngine:
    def __init__(self, confidence_config: ConfidenceConfig, model_scorer: ModelScorer):
        self.confidence_config = confidence_config
        self.model_scorer = model_scorer
        self.indirect_confidence_method  = IndirectConfidenceMethod(self.model_scorer)
        self.verbal_confidence_method = VerbalConfidenceMethod(self.model_scorer)

    def compute_confidence(self, parsed_output: ParsedOutputGeneration) -> ConfidenceScores:
        # answer_token_probs is [T_answer, vocab_size]. For each answer-position
        # row, take the argmax token and its probability.
        probs = parsed_output.answer_token_probs.detach().cpu()
        tokenizer = self.model_scorer.model.tokenizer
        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        if probs.numel() > 0:
            max_probs, max_ids = torch.max(probs, dim=-1)
            # Entropy at each answer token (row) -- compute over all vocab probs per row
            # Entropy H(p) = -sum_i p_i log(p_i)
            probs_for_entropy = probs.double()
            logprobs = torch.nan_to_num(probs_for_entropy.log(), neginf=-99)
            entropies = -torch.sum(probs_for_entropy * logprobs, dim=-1)
            for i, (prob, tok_id) in enumerate(zip(max_probs.tolist(), max_ids.tolist())):
                token_str = tokenizer.decode([int(tok_id)])
                answer_probabilities.append({token_str: float(prob)})
                answer_entropy.append({token_str: float(entropies[i].item())})


        indirect_scores = self.indirect_confidence_method.compute_confidence(parsed_output)
        verbal_scores = self.verbal_confidence_method.compute_confidence(parsed_output)
        return ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
        )

