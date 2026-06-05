
import math

import torch

from openai.types.chat.chat_completion_token_logprob import TopLogprob

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

    def _compute_confidence_tensor_probs(self, parsed_output: ParsedOutputGeneration) -> ConfidenceScores:
        # answer_token_probs is [T_answer, vocab_size]. For each answer-position
        # row, take the actually-sampled token (parsed_output.answer_token_ids)
        # and gather its probability.
        probs = parsed_output.answer_token_probs.detach().cpu()
        tokenizer = self.model_scorer.model.tokenizer
        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        answer_top20_probabilities: list[dict[str, float]] | None = [] if self.confidence_config.debug_top20 else None

        # experimental - stats for scores
        answer_score_probs: list[dict[str, float]] = []
        answer_score_entropy: list[dict[str, float]] = []
        answer_token_score_probs: torch.Tensor | None = parsed_output.answer_token_score_probs
        if answer_token_score_probs is not None:
            ids = parsed_output.answer_token_ids.detach().cpu().long()
            answer_token_score_probs = answer_token_score_probs.detach().cpu()
            selected_score_probs = answer_token_score_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            score_probs_for_entropy = answer_token_score_probs.double()
            logprobs = torch.nan_to_num(score_probs_for_entropy.log(), neginf=-99)
            entropies = -torch.sum(score_probs_for_entropy * logprobs, dim=-1)
            for i, (prob, tok_id) in enumerate(zip(selected_score_probs.tolist(), ids.tolist())):
                token_str = tokenizer.decode([int(tok_id)])
                answer_score_probs.append({token_str: float(prob)})
                answer_score_entropy.append({token_str: float(entropies[i].item())})

        if probs.numel() > 0:
            ids = parsed_output.answer_token_ids.detach().cpu().long()
            selected_probs = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)

            if self.confidence_config.debug_top20:
                top_k = min(20, probs.shape[-1])
                top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
                for token_probs, token_ids in zip(top_probs.tolist(), top_ids.tolist()):
                    answer_top20_probabilities.append({
                        tokenizer.decode([int(tok_id)]): float(prob)
                        for tok_id, prob in zip(token_ids, token_probs)
                    })

            # Entropy at each answer token (row) -- compute over all vocab probs per row
            # Entropy H(p) = -sum_i p_i log(p_i)
            probs_for_entropy = probs.double()
            logprobs = torch.nan_to_num(probs_for_entropy.log(), neginf=-99)
            entropies = -torch.sum(probs_for_entropy * logprobs, dim=-1)
            for i, (prob, tok_id) in enumerate(zip(selected_probs.tolist(), ids.tolist())):
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
            answer_score_probabilities=answer_score_probs,
            answer_score_entropy=answer_score_entropy,
            debug_answer_top20_probabilities=answer_top20_probabilities,
        )


    def _compute_confidence_list_probs(self, parsed_output: ParsedOutputGeneration) -> ConfidenceScores:
        # answer_token_probs is list[list[TopLogprob]]: per answer-token position,
        # the top-N (token, logprob) pairs returned by the API. answer_token_ids
        # is list[str] of the actually-sampled token strings (already decoded).
        # We only see the top-N distribution, not the full vocab.
        top_logprobs_per_pos: list[list[TopLogprob]] = parsed_output.answer_token_probs
        sampled_tokens: list[str] = parsed_output.answer_token_ids




        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        answer_top20_probabilities: list[dict[str, float]] | None = [] if self.confidence_config.debug_top20 else None

        for token_str, top_logprobs in zip(sampled_tokens, top_logprobs_per_pos):
            scores = {lp.token: lp.logprob for lp in top_logprobs}
            # Sampled token's probability: 0.0 if it fell outside the top-N.
            sampled_logprob = scores.get(token_str)
            sampled_prob = math.exp(sampled_logprob) if sampled_logprob is not None else 0.0

            # Entropy over the available top-N only (no full vocab from the API).
            top_probs = [math.exp(lp) for lp in scores.values()]
            entropy = -sum(p * math.log(p) for p in top_probs if p > 0.0)

            answer_probabilities.append({token_str: sampled_prob})
            answer_entropy.append({token_str: entropy})

            if self.confidence_config.debug_top20:
                answer_top20_probabilities.append({tok: math.exp(lp) for tok, lp in scores.items()})

        # API models don't populate answer_token_score_probs.
        answer_score_probs: list[dict[str, float]] = []
        answer_score_entropy: list[dict[str, float]] = []

        indirect_scores = self.indirect_confidence_method.compute_confidence(parsed_output)
        verbal_scores = self.verbal_confidence_method.compute_confidence(parsed_output)
        return ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
            answer_score_probabilities=answer_score_probs,
            answer_score_entropy=answer_score_entropy,
            debug_answer_top20_probabilities=answer_top20_probabilities,
        )




    def compute_confidence(self, parsed_output: ParsedOutputGeneration) -> ConfidenceScores:
        if isinstance(parsed_output.answer_token_probs, torch.Tensor):
            return self._compute_confidence_tensor_probs(parsed_output)
        elif isinstance(parsed_output.answer_token_probs, list):
            return self._compute_confidence_list_probs(parsed_output)
        else:
            raise ValueError(f"Unsupported answer_token_probs type: {type(parsed_output.answer_token_probs)}")
