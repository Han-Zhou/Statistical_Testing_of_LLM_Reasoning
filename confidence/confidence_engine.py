
import asyncio
import math
import time
from typing import Any

import torch

from openai.types.chat.chat_completion_token_logprob import TopLogprob

from config import ConfidenceConfig
from domain import ParsedOutputGeneration, ConfidenceScores, ConfidenceTime
from confidence.indirect import IndirectConfidenceMethod
from confidence.verbal import VerbalConfidenceMethod
from models.adapters import ModelScorer






class ConfidenceEngine:
    def __init__(self, confidence_config: ConfidenceConfig, model_scorer: ModelScorer):
        self.confidence_config = confidence_config
        self.model_scorer = model_scorer
        self.indirect_confidence_method  = IndirectConfidenceMethod(self.model_scorer)
        self.verbal_confidence_method = VerbalConfidenceMethod(self.model_scorer)

    def time_stamp(self) -> float:
        """Return a timestamp in seconds, optionally synchronizing CUDA first."""
        if self.confidence_config.cuda_sync_for_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


    def _compute_confidence_tensor_probs(self, parsed_output: ParsedOutputGeneration) -> tuple[ConfidenceScores, ConfidenceTime]:
        # answer_token_probs is [T_answer, vocab_size]. For each answer-position
        # row, take the actually-sampled token (parsed_output.answer_token_ids)
        # and gather its probability.
        probs = parsed_output.answer_token_probs.detach().cpu()
        tokenizer = self.model_scorer.model.tokenizer
        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        scores_debug: dict[str, Any] = {}

        # experimental - stats for scores
        answer_score_probs: list[dict[str, float]] = []
        answer_score_entropy: list[dict[str, float]] = []
        answer_token_score_probs: torch.Tensor | None = parsed_output.answer_token_score_probs
        answer_prob_time = 0.0
        answer_ent_time = 0.0
        answer_score_prob_time = 0.0
        answer_score_ent_time = 0.0
        if answer_token_score_probs is not None:
            ids = parsed_output.answer_token_ids.detach().cpu().long()
            answer_token_score_probs = answer_token_score_probs.detach().cpu()
            t1 = self.time_stamp()
            selected_score_probs = answer_token_score_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            t2 = self.time_stamp()
            score_probs_for_entropy = answer_token_score_probs.double()
            logprobs = torch.nan_to_num(score_probs_for_entropy.log(), neginf=-99)
            entropies = -torch.sum(score_probs_for_entropy * logprobs, dim=-1)
            t3 = self.time_stamp()
            answer_score_prob_time += t2 - t1
            answer_score_ent_time += t3 - t2
            for i, (prob, tok_id) in enumerate(zip(selected_score_probs.tolist(), ids.tolist())):
                token_str = tokenizer.decode([int(tok_id)])
                answer_score_probs.append({token_str: float(prob)})
                answer_score_entropy.append({token_str: float(entropies[i].item())})

        if probs.numel() > 0:
            ids = parsed_output.answer_token_ids.detach().cpu().long()
            t1 = self.time_stamp()
            selected_probs = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            t2 = self.time_stamp()

            # Entropy at each answer token (row) -- compute over all vocab probs per row
            # Entropy H(p) = -sum_i p_i log(p_i)
            probs_for_entropy = probs.double()
            logprobs = torch.nan_to_num(probs_for_entropy.log(), neginf=-99)
            entropies = -torch.sum(probs_for_entropy * logprobs, dim=-1)
            t3 = self.time_stamp()

            if self.confidence_config.debug_top20:
                top_k = min(20, probs.shape[-1])
                top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
                scores_debug["answer_top20_probabilities"] = [
                    {tokenizer.decode([int(tok_id)]): float(prob)
                     for tok_id, prob in zip(token_ids, token_probs)}
                    for token_probs, token_ids in zip(top_probs.tolist(), top_ids.tolist())
                ]

            answer_prob_time += t2 - t1
            answer_ent_time += t3 - t2
            for i, (prob, tok_id) in enumerate(zip(selected_probs.tolist(), ids.tolist())):
                token_str = tokenizer.decode([int(tok_id)])
                answer_probabilities.append({token_str: float(prob)})
                answer_entropy.append({token_str: float(entropies[i].item())})

        t1 = self.time_stamp()
        indirect_scores, debug_indirect = self.indirect_confidence_method.compute_confidence(parsed_output)
        t2 = self.time_stamp()
        verbal_scores, debug_verbal = self.verbal_confidence_method.compute_confidence(parsed_output)
        t3 = self.time_stamp()

        indirect_time = t2 - t1
        verbconf_time = t3 - t2
        time_debug = {
            "indirect_time_debug": debug_indirect,
            "verbconf_time_debug": debug_verbal,
        }

        confidence_scores = ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
            answer_score_probabilities=answer_score_probs,
            answer_score_entropy=answer_score_entropy,
            debug=scores_debug or None,
        )
        confidence_time = ConfidenceTime(
            answer_prob_time=answer_prob_time,
            answer_ent_time=answer_ent_time,
            indirect_time=indirect_time,
            verbconf_time=verbconf_time,
            answer_score_prob_time=answer_score_prob_time,
            answer_score_ent_time=answer_score_ent_time,
            debug=time_debug or None,
        )
        return confidence_scores, confidence_time
    
    

    @staticmethod
    def _normalize_openai_logprobs(top_logprobs: list[TopLogprob]) -> dict[str, float]:
        """OpenAI TopLogprob list → {token_str: probability}."""
        return {lp.token: math.exp(lp.logprob) for lp in top_logprobs}

    def _normalize_vllm_logprobs(self, logprob_dict: dict[int, float]) -> dict[str, float]:
        """vLLM dict[token_id, probability] → {token_str: probability}."""
        tokenizer = self.model_scorer.model.tokenizer
        return {tokenizer.decode([tok_id]): prob for tok_id, prob in logprob_dict.items()}

    def _compute_confidence_list_probs(self, parsed_output: ParsedOutputGeneration) -> tuple[ConfidenceScores, ConfidenceTime]:
        sampled_tokens: list[str] = parsed_output.answer_token_ids
        raw_probs_per_pos = parsed_output.answer_token_probs

        is_vllm = len(raw_probs_per_pos) > 0 and isinstance(raw_probs_per_pos[0], dict)

        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        scores_debug: dict[str, Any] = {}
        answer_top20_probabilities: list[dict[str, float]] | None = (
            [] if self.confidence_config.debug_top20 else None
        )
        answer_prob_time = 0.0
        answer_ent_time = 0.0

        for token_str, raw_entry in zip(sampled_tokens, raw_probs_per_pos):
            t1 = self.time_stamp()
            if is_vllm:
                probs_by_token = self._normalize_vllm_logprobs(raw_entry)
            else:
                probs_by_token = self._normalize_openai_logprobs(raw_entry)

            sampled_prob = probs_by_token.get(token_str, 0.0)
            t2 = self.time_stamp()
            top_probs = list(probs_by_token.values())
            entropy = -sum(p * math.log(p) for p in top_probs if p > 0.0)
            t3 = self.time_stamp()
            answer_prob_time += t2 - t1
            answer_ent_time += t3 - t2

            answer_probabilities.append({token_str: sampled_prob})
            answer_entropy.append({token_str: entropy})

            if answer_top20_probabilities is not None:
                answer_top20_probabilities.append(probs_by_token)

        if answer_top20_probabilities is not None:
            scores_debug["answer_top20_probabilities"] = answer_top20_probabilities

        # API models don't populate answer_token_score_probs.
        answer_score_probs: list[dict[str, float]] = []
        answer_score_entropy: list[dict[str, float]] = []

        t1 = self.time_stamp()
        indirect_scores, debug_indirect = self.indirect_confidence_method.compute_confidence(parsed_output)
        t2 = self.time_stamp()
        verbal_scores, debug_verbal = self.verbal_confidence_method.compute_confidence(parsed_output)
        t3 = self.time_stamp()

        indirect_time = t2 - t1
        verbconf_time = t3 - t2
        time_debug = {
            "indirect_time_debug": debug_indirect,
            "verbconf_time_debug": debug_verbal,
        }

        confidence_scores = ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
            answer_score_probabilities=answer_score_probs,
            answer_score_entropy=answer_score_entropy,
            debug=scores_debug or None,
        )
        confidence_time = ConfidenceTime(
            answer_prob_time=answer_prob_time,
            answer_ent_time=answer_ent_time,
            indirect_time=indirect_time,
            verbconf_time=verbconf_time,
            answer_score_prob_time=0.0,
            answer_score_ent_time=0.0,
            debug=time_debug,
        )
        return ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
            answer_score_probabilities=answer_score_probs,
            answer_score_entropy=answer_score_entropy,
            debug=scores_debug or None,
        ), confidence_time




    def compute_confidence(self, parsed_output: ParsedOutputGeneration) -> tuple[ConfidenceScores, ConfidenceTime]:
        if isinstance(parsed_output.answer_token_probs, torch.Tensor):
            return self._compute_confidence_tensor_probs(parsed_output)
        elif isinstance(parsed_output.answer_token_probs, list):
            return self._compute_confidence_list_probs(parsed_output)
        else:
            raise ValueError(f"Unsupported answer_token_probs type: {type(parsed_output.answer_token_probs)}")

    async def _compute_confidence_list_probs_async(
        self,
        parsed_output: ParsedOutputGeneration,
        semaphore: asyncio.Semaphore,
    ) -> tuple[ConfidenceScores, ConfidenceTime]:
        """Score one API output while limiting each remote confidence request."""
        sampled_tokens: list[str] = parsed_output.answer_token_ids
        raw_probs_per_pos = parsed_output.answer_token_probs
        is_vllm = len(raw_probs_per_pos) > 0 and isinstance(raw_probs_per_pos[0], dict)

        answer_probabilities: list[dict[str, float]] = []
        answer_entropy: list[dict[str, float]] = []
        answer_top20_probabilities = [] if self.confidence_config.debug_top20 else None
        answer_prob_time = 0.0
        answer_ent_time = 0.0

        for token_str, raw_entry in zip(sampled_tokens, raw_probs_per_pos):
            t1 = self.time_stamp()
            if is_vllm:
                probs_by_token = self._normalize_vllm_logprobs(raw_entry)
            else:
                probs_by_token = self._normalize_openai_logprobs(raw_entry)
            sampled_prob = probs_by_token.get(token_str, 0.0)
            t2 = self.time_stamp()
            top_probs = list(probs_by_token.values())
            entropy = -sum(p * math.log(p) for p in top_probs if p > 0.0)
            t3 = self.time_stamp()
            answer_prob_time += t2 - t1
            answer_ent_time += t3 - t2
            answer_probabilities.append({token_str: sampled_prob})
            answer_entropy.append({token_str: entropy})
            if answer_top20_probabilities is not None:
                answer_top20_probabilities.append(probs_by_token)

        async with semaphore:
            t1 = self.time_stamp()
            indirect_scores, debug_indirect = (
                await self.indirect_confidence_method.compute_confidence_async(parsed_output)
            )
            t2 = self.time_stamp()
        async with semaphore:
            t3 = self.time_stamp()
            verbal_scores, debug_verbal = (
                await self.verbal_confidence_method.compute_confidence_async(parsed_output)
            )
            t4 = self.time_stamp()

        scores_debug = None
        if answer_top20_probabilities is not None:
            scores_debug = {"answer_top20_probabilities": answer_top20_probabilities}
        scores = ConfidenceScores(
            answer_probabilities=answer_probabilities,
            answer_entropy=answer_entropy,
            indirect_probabilities=indirect_scores,
            verbconf_probabilities=verbal_scores,
            answer_score_probabilities=[],
            answer_score_entropy=[],
            debug=scores_debug,
        )
        timings = ConfidenceTime(
            answer_prob_time=answer_prob_time,
            answer_ent_time=answer_ent_time,
            indirect_time=t2 - t1,
            verbconf_time=t4 - t3,
            answer_score_prob_time=0.0,
            answer_score_ent_time=0.0,
            debug={
                "indirect_time_debug": debug_indirect,
                "verbconf_time_debug": debug_verbal,
            },
        )
        return scores, timings

    async def compute_confidence_batch_async(
        self,
        parsed_outputs: list[ParsedOutputGeneration],
        concurrency: int,
    ) -> list[tuple[ConfidenceScores, ConfidenceTime]]:
        """Concurrently score API outputs, preserving their input order."""
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if any(not isinstance(output.answer_token_probs, list) for output in parsed_outputs):
            raise ValueError("Async confidence scoring supports API/list probabilities only")
        semaphore = asyncio.Semaphore(concurrency)
        return list(await asyncio.gather(*(
            self._compute_confidence_list_probs_async(output, semaphore)
            for output in parsed_outputs
        )))

    @staticmethod
    def _batch_token_score(token_scores, token_id: int) -> torch.Tensor:
        """Read one token score from either HF dense logits or vLLM logprobs."""
        if isinstance(token_scores, dict):
            value = token_scores.get(token_id, -100.0)
            # vLLM dictionaries contain Logprob records; tests and other
            # backends may provide plain numeric values.
            value = getattr(value, "logprob", value)
            return torch.as_tensor(value).detach().cpu()
        return token_scores[token_id].detach().cpu()


    def compute_confidence_batch(
        self,
        parsed_outputs: list[ParsedOutputGeneration],
        shared_cache: 'CacheBundle | None' = None,
    ) -> list[tuple[ConfidenceScores, ConfidenceTime]]:
        """Batched confidence scoring: runs all indirect + verbal forward passes
        for N samples in a single batched forward call.
        If shared_cache is provided, uses cache-replicated batching over delta tokens only."""
        from models.adapters.registry import ANSWER_TOKENS

        if not parsed_outputs:
            return []
        tok = self.model_scorer.model.tokenizer

        # Build prompts for indirect and verbal, N each → 2N total
        indirect_prompts = []
        verbal_prompts = []
        for po in parsed_outputs:
            indirect_tail = self.indirect_confidence_method.tail_prompt(po.final_answer)
            verbal_tail = self.verbal_confidence_method.tail_prompt(po.final_answer)
            if po.input_messages is not None:
                indirect_prompt = po.input_messages + [
                    {"role": "assistant", "content": po.text_cot + indirect_tail}
                ]
                verbal_prompt = po.input_messages + [
                    {"role": "assistant", "content": po.text_cot + verbal_tail}
                ]
                # render via tokenizer chat template
                indirect_prompts.append(
                    tok.apply_chat_template(indirect_prompt, tokenize=False)
                )
                verbal_prompts.append(
                    tok.apply_chat_template(verbal_prompt, tokenize=False)
                )
            else:
                indirect_prompts.append(po.text_question + po.text_cot + indirect_tail)
                verbal_prompts.append(po.text_question + po.text_cot + verbal_tail)

        # Two batched forward passes: one for indirect, one for verbal
        t0 = self.time_stamp()
        indirect_last_logits = self.model_scorer.forward_batch_confidence(indirect_prompts, shared_cache=shared_cache)
        t1 = self.time_stamp()
        verbal_last_logits = self.model_scorer.forward_batch_confidence(verbal_prompts, shared_cache=shared_cache)
        t2 = self.time_stamp()
        if (
            len(indirect_last_logits) != len(parsed_outputs)
            or len(verbal_last_logits) != len(parsed_outputs)
        ):
            raise RuntimeError("Batch confidence scoring returned an unexpected number of outputs")
        indirect_time = t1 - t0
        verbconf_time = t2 - t1

        # Extract indirect and verbal results
        true_id = tok(ANSWER_TOKENS[' True'][0], add_special_tokens=False).input_ids[0]
        false_id = tok(ANSWER_TOKENS[' False'][0], add_special_tokens=False).input_ids[0]
        verbal_token_ids = [
            tok(s, add_special_tokens=False).input_ids[0]
            for s in ANSWER_TOKENS['llama_verbal_confidence']
        ]
        verbal_token_strs = list(ANSWER_TOKENS['llama_verbal_confidence'])

        results = []
        for i, po in enumerate(parsed_outputs):
            # Indirect: logits at position i
            indirect_logits = indirect_last_logits[i]
            scorer_output_indirect = {
                'True': self._batch_token_score(indirect_logits, true_id),
                'False': self._batch_token_score(indirect_logits, false_id),
            }
            indirect_scores = self.indirect_confidence_method.extract(scorer_output_indirect)

            # Verbal: logits at position i
            verbal_logits = verbal_last_logits[i]
            scorer_output_verbal = {
                s: self._batch_token_score(verbal_logits, tid)
                for s, tid in zip(verbal_token_strs, verbal_token_ids)
            }
            verbal_scores = self.verbal_confidence_method.extract(scorer_output_verbal)

            # Compute answer token probs/entropy (same as serial path)
            answer_probabilities: list[dict[str, float]] = []
            answer_entropy: list[dict[str, float]] = []
            scores_debug: dict = {}
            answer_score_probs: list[dict[str, float]] = []
            answer_score_entropy: list[dict[str, float]] = []
            answer_prob_time = 0.0
            answer_ent_time = 0.0
            answer_score_prob_time = 0.0
            answer_score_ent_time = 0.0

            answer_token_score_probs = po.answer_token_score_probs
            if answer_token_score_probs is not None:
                ids = po.answer_token_ids.detach().cpu().long()
                answer_token_score_probs = answer_token_score_probs.detach().cpu()
                t_s1 = self.time_stamp()
                selected_score_probs = answer_token_score_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                t_s2 = self.time_stamp()
                score_probs_for_entropy = answer_token_score_probs.double()
                logprobs = torch.nan_to_num(score_probs_for_entropy.log(), neginf=-99)
                entropies = -torch.sum(score_probs_for_entropy * logprobs, dim=-1)
                t_s3 = self.time_stamp()
                answer_score_prob_time += t_s2 - t_s1
                answer_score_ent_time += t_s3 - t_s2
                for j, (prob, tok_id) in enumerate(zip(selected_score_probs.tolist(), ids.tolist())):
                    token_str = tok.decode([int(tok_id)])
                    answer_score_probs.append({token_str: float(prob)})
                    answer_score_entropy.append({token_str: float(entropies[j].item())})

            if isinstance(po.answer_token_probs, torch.Tensor):
                probs = po.answer_token_probs.detach().cpu()
                if probs.numel() > 0:
                    ids = po.answer_token_ids.detach().cpu().long()
                    t_a1 = self.time_stamp()
                    selected_probs = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                    t_a2 = self.time_stamp()
                    probs_for_entropy = probs.double()
                    logprobs_e = torch.nan_to_num(probs_for_entropy.log(), neginf=-99)
                    entropies_e = -torch.sum(probs_for_entropy * logprobs_e, dim=-1)
                    t_a3 = self.time_stamp()
                    answer_prob_time += t_a2 - t_a1
                    answer_ent_time += t_a3 - t_a2

                    if self.confidence_config.debug_top20:
                        top_k = min(20, probs.shape[-1])
                        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
                        scores_debug["answer_top20_probabilities"] = [
                            {tok.decode([int(tok_id)]): float(prob)
                             for tok_id, prob in zip(token_ids, token_probs)}
                            for token_probs, token_ids in zip(top_probs.tolist(), top_ids.tolist())
                        ]

                    for j, (prob, tok_id) in enumerate(zip(selected_probs.tolist(), ids.tolist())):
                        token_str = tok.decode([int(tok_id)])
                        answer_probabilities.append({token_str: float(prob)})
                        answer_entropy.append({token_str: float(entropies_e[j].item())})
            elif isinstance(po.answer_token_probs, list):
                answer_top20_probabilities = (
                    [] if self.confidence_config.debug_top20 else None
                )
                raw_probs_per_pos = po.answer_token_probs
                is_vllm = bool(raw_probs_per_pos) and isinstance(raw_probs_per_pos[0], dict)
                for token_str, raw_entry in zip(po.answer_token_ids, raw_probs_per_pos):
                    t_a1 = self.time_stamp()
                    if is_vllm:
                        probs_by_token = self._normalize_vllm_logprobs(raw_entry)
                    else:
                        probs_by_token = self._normalize_openai_logprobs(raw_entry)
                    sampled_prob = probs_by_token.get(token_str, 0.0)
                    t_a2 = self.time_stamp()
                    entropy = -sum(
                        prob * math.log(prob)
                        for prob in probs_by_token.values()
                        if prob > 0.0
                    )
                    t_a3 = self.time_stamp()
                    answer_prob_time += t_a2 - t_a1
                    answer_ent_time += t_a3 - t_a2
                    answer_probabilities.append({token_str: sampled_prob})
                    answer_entropy.append({token_str: entropy})
                    if answer_top20_probabilities is not None:
                        answer_top20_probabilities.append(probs_by_token)
                if answer_top20_probabilities is not None:
                    scores_debug["answer_top20_probabilities"] = answer_top20_probabilities
            else:
                raise ValueError(
                    "Unsupported answer_token_probs type in batch: "
                    f"{type(po.answer_token_probs)}"
                )

            confidence_scores = ConfidenceScores(
                answer_probabilities=answer_probabilities,
                answer_entropy=answer_entropy,
                indirect_probabilities=indirect_scores,
                verbconf_probabilities=verbal_scores,
                answer_score_probabilities=answer_score_probs,
                answer_score_entropy=answer_score_entropy,
                debug=scores_debug or None,
            )
            confidence_time = ConfidenceTime(
                answer_prob_time=answer_prob_time,
                answer_ent_time=answer_ent_time,
                indirect_time=indirect_time,
                verbconf_time=verbconf_time,
                answer_score_prob_time=answer_score_prob_time,
                answer_score_ent_time=answer_score_ent_time,
                debug=None,
            )
            results.append((confidence_scores, confidence_time))

        return results
