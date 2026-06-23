import re
import copy
import time
from types import SimpleNamespace

from typing import Optional, Tuple

from transformers.utils import ModelOutput

from domain import LLMOutput, ParsedOutputGeneration, KVCache, CacheBundle, AnswerSpan, ScorerOutput


from models.adapters.base import ModelAdapter, ModelScorer
from models.core_models.llm import LLM
from models.adapters.registry import MODEL_ATTENTION_IMPLEMENTATION_REGISTRY, ANSWER_TOKENS
from models.adapters.shared_utils import _locate_answer_span, _char_to_token_idx



import torch.nn.functional as F
import torch



LLAMA_STOP_STRINGS = [
    "\nAnswer:",
    "\nFinal Answer",
    "\nFinal answer",
    "\nThe final answer",
    "\nThe answer",
    "\nTherefore the answer",
    "\nTherefore the final answer",
]
# Extend stop strings with "\n\n{letter}\n" and "\n\n({letter})\n" for each letter in LETTERS
LETTERS = "ABCD"
LLAMA_STOP_STRINGS.extend(f"\n\n{letter}\n" for letter in LETTERS)
LLAMA_STOP_STRINGS.extend(f"\n\n({letter})\n" for letter in LETTERS)




class LlamaScorer(ModelScorer):
    def __init__(self, model: LLM):
        self.model = model

    def forward_indirect(self, prompt: str, whole_cache: CacheBundle) -> tuple[ScorerOutput, dict[str, float]]:
        """
        forward_indirect runs a forward pass on the prompt with indirect suffix, and returns the logitsfor the indirect tokens. These are 'True' and 'False' tokens generated last
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = self.model.align_cache(whole_cache, prompt)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        outputs = self.model.forward(prompt, cache=cache)
        last_logits = outputs.logits[0, -1, :]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tok = self.model.tokenizer
        true_id = tok(ANSWER_TOKENS[' True'][0], add_special_tokens=False).input_ids[0]
        false_id = tok(ANSWER_TOKENS[' False'][0], add_special_tokens=False).input_ids[0]
        t3 = time.perf_counter()

        # de-tokenize the aligned cache so we can see what prefix got reused.
        # `cache` is the LCP-cropped KVCache; its length is the number of reused
        # tokens, which are exactly the first N ids of whole_cache.input_ids.
        reused_num_tokens = 0 if cache is None else cache.get_seq_length()
        reused_tokens_text = tok.decode(
            whole_cache.input_ids[:reused_num_tokens], skip_special_tokens=False
        )

        debug_info = {
            "align_cache_time": t1 - t0,
            "forward_time": t2 - t1,
            "tokenizer_time": t3 - t2,
            "reused_num_tokens": reused_num_tokens,
            "reused_tokens_text": reused_tokens_text,
        }

        return ({
            'True': last_logits[true_id].detach().cpu(),
            'False': last_logits[false_id].detach().cpu(),
        }, debug_info)


    def forward_verbal(self, prompt: str, whole_cache: CacheBundle) -> tuple[ScorerOutput, dict[str, float]]:
        """
        forward_verbal runs a forward pass on the prompt with verbal suffix, and returns the logits for the verbal tokens. These are the tokens generated last.
        For llama, we are lucky such that every integer [0, 100] has its own token, so we only need one forward pass
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = self.model.align_cache(whole_cache, prompt)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        outputs = self.model.forward(prompt, cache=cache)
        last_logits = outputs.logits[0, -1, :]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tok = self.model.tokenizer

        # de-tokenize the aligned cache so we can see what prefix got reused.
        # `cache` is the LCP-cropped KVCache; its length is the number of reused
        # tokens, which are exactly the first N ids of whole_cache.input_ids.
        reused_num_tokens = 0 if cache is None else cache.get_seq_length()
        reused_tokens_text = tok.decode(
            whole_cache.input_ids[:reused_num_tokens], skip_special_tokens=False
        )

        debug_info = {
            "align_cache_time": t1 - t0,
            "forward_time": t2 - t1,
            "reused_num_tokens": reused_num_tokens,
            "reused_tokens_text": reused_tokens_text,
        }
        return ({
            s: last_logits[tok(s, add_special_tokens=False).input_ids[0]].detach().cpu()
            for s in ANSWER_TOKENS['llama_verbal_confidence']
        }, debug_info)


    def forward_batch_confidence(
        self,
        prompts: list[str],
        shared_cache: CacheBundle | None = None,
    ) -> list[torch.Tensor]:
        """Batched forward pass returning last-token logits for each prompt.
        If shared_cache is provided (and debug_nocache is off), uses cache-replicated batching.
        Otherwise falls back to full-prompt left-padded batching."""
        if shared_cache is not None and shared_cache.cache is not None and not self.model.debug_nocache:
            # Compute the delta text for each prompt: strip the shared prefix
            prefix_text = self.model.tokenizer.decode(
                shared_cache.input_ids, skip_special_tokens=False
            )
            prefix_len = len(prefix_text)
            delta_texts = []
            for p in prompts:
                if p.startswith(prefix_text):
                    delta_texts.append(p[prefix_len:])
                else:
                    # Fallback: can't split cleanly, use full prompts without cache
                    all_last_logits = self.model.forward_batch_last_logits(prompts)
                    return [all_last_logits[i] for i in range(all_last_logits.shape[0])]
            cache_seq_len = shared_cache.cache.get_seq_length()
            all_last_logits = self.model.forward_batch_last_logits_with_cache(
                delta_texts=delta_texts,
                cache=shared_cache.cache,
                cache_seq_len=cache_seq_len,
            )
        else:
            all_last_logits = self.model.forward_batch_last_logits(prompts)
        return [all_last_logits[i] for i in range(all_last_logits.shape[0])]



    # def forward_continuations(
    #     self,
    #     continuation_texts: list[str],
    #     cache: KVCache,
    #     last_prompt_token_id: int,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Batched forward pass over N candidate continuations conditioned on `cache`.
        
    #     Returns:
    #         logits:  [N, T_max, vocab_size] where logits[n, i] predicts the i-th
    #                 candidate token of continuation n (i.e., aligned with the
    #                 candidate tokens themselves, not shifted by one).
    #         lengths: [N] true token length of each continuation (for masking padding).
    #     """
    #     tok = self.model.tokenizer
    #     pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    #     cand_ids = [tok(c, add_special_tokens=False).input_ids for c in continuation_texts]
    #     lengths = torch.tensor([len(ids) for ids in cand_ids])
    #     T_max = int(lengths.max().item())
    #     N = len(cand_ids)

    #     batched = torch.full((N, T_max + 1), pad_id, dtype=torch.long)
    #     batched[:, 0] = last_prompt_token_id
    #     for i, ids in enumerate(cand_ids):
    #         batched[i, 1:1 + len(ids)] = torch.tensor(ids)
    #     batched = batched.to(self.model.device)

    #     short_cache = self._slice_cache(cache, 0, -1)
    #     batched_cache = self._replicate_cache(short_cache, N) if N > 1 else short_cache

    #     with torch.no_grad():
    #         out = self.model.model(
    #             input_ids=batched,
    #             past_key_values=batched_cache,
    #             use_cache=False,
    #         )
    #     # out.logits: [N, T_max+1, vocab]
    #     # out.logits[:, :-1] aligns with the candidate token positions
    #     # out.logits[:, -1]  is the next-token-after-continuation distribution
    #     return out.logits.detach(), lengths





class LlamaAdapter(ModelAdapter):
    def __init__(self, debug_nocache: bool = False):
        attention_implementation = MODEL_ATTENTION_IMPLEMENTATION_REGISTRY.get("llama")
        self.model = LLM(model_name="llama", attention_implementation=attention_implementation, debug_nocache=debug_nocache)
        self.model_scorer = LlamaScorer(self.model)


    def _strip_trailing_special_token(self, text: str, cache: KVCache) -> tuple[str, KVCache]:
        for special_tok in self.model.tokenizer.all_special_tokens:
            if text.endswith(special_tok):
                special_token_ids = self.model.tokenizer(
                    special_tok,
                    add_special_tokens=False,
                ).input_ids
                cache.crop(cache.get_seq_length() - len(special_token_ids))
                return text[:-len(special_tok)], cache
        return text, cache



    def _slice_cache(self, cache: KVCache, start: int, end: int) -> KVCache:
        sliced = copy.deepcopy(cache)
        for layer in sliced.layers:
            k = layer.keys
            v = layer.values
            if k is not None and k.dim() == 4:
                layer.keys = k[:, :, start:end, :]
                layer.values = v[:, :, start:end, :]
        if hasattr(sliced, '_seen_tokens'):
            sliced._seen_tokens = end - start
        return sliced



    def _extract_cot(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], cache: KVCache | None, sequence_ids: torch.Tensor, cot_start_idx: int, answer_span: AnswerSpan | None) -> tuple[list[str], str, str, str, CacheBundle | None, CacheBundle | None]:
        # text_cot_with_answer contains basically everything after the "assistant" header
        text_cot_with_answer = output_text[cot_start_idx:]
        text_question = output_text[:cot_start_idx]

        # text_cot contains only the CoT part, without the final answer
        if answer_span is not None:
            text_cot = output_text[cot_start_idx:answer_span.char_answer_sentence_start]
        else:
            text_cot = text_cot_with_answer

        # we extract the cot_steps out of text_cot
        _STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)
        parts = _STEP_MARKER_RE.split(text_cot)
        if len(parts) > 1:
            steps = []
            preamble = parts[0].strip()
            if preamble:
                steps.append(preamble)
            for i in range(1, len(parts) - 1, 2):
                steps.append((parts[i] + parts[i + 1]).strip())
            cot_steps = [s for s in steps if s]
        # No "Step N:" markers — fall back to blank-line, then line splits
        else:
            by_blank = [s.strip() for s in re.split(r"\n{2,}", text_cot) if s.strip()]
            if len(by_blank) > 1:
                cot_steps = by_blank
            else:
                cot_steps = [s.strip() for s in text_cot.splitlines() if s.strip()]

        if cache is None:
            return cot_steps, text_question, text_cot, text_cot_with_answer, None, None

        # Split the cache at the assistant-header boundary. CacheBundle.cropped_to
        # handles BPE drift downstream via longest_common_prefix, so we don't need
        # to back off by a token here.
        cot_start_token_idx = _char_to_token_idx(self, cot_start_idx, offset_mappings)
        question_kv = copy.deepcopy(cache)
        question_kv.crop(cot_start_token_idx)
        question_cache = CacheBundle(
            cache=question_kv,
            input_ids=sequence_ids[:cot_start_token_idx].detach().cpu().clone(),
        )
        whole_cache = CacheBundle(
            cache=copy.deepcopy(cache),
            input_ids=sequence_ids[:cache.get_seq_length()].detach().cpu().clone(),
        )

        return cot_steps, text_question, text_cot, text_cot_with_answer, whole_cache, question_cache


    def _extract_answer_and_probs(
        self, 
        output_text: str, 
        output_tokens: list[str], 
        offset_mappings: list[tuple[int, int]], 
        all_probs: torch.Tensor, 
        all_score_probs: torch.Tensor | None, 
        sequence_ids: torch.Tensor, 
        answer_span: AnswerSpan | None
    ) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if answer_span is None:
            return "", torch.tensor([]), torch.tensor([], dtype=torch.long), None

        final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()

        # these token indices are absolute
        answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
        answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)

        # all_probs is relative to the start of generation, so we need to shift these token indices by the number of question tokens
        num_question_tokens = len(output_tokens) - all_probs.shape[0]
        answer_token_probs = all_probs[
            answer_start_token_idx - num_question_tokens
            :answer_end_token_idx - num_question_tokens
        ]
        if all_score_probs is None:
            answer_token_score_probs = None
        else:
            num_score_question_tokens = len(output_tokens) - all_score_probs.shape[0]
            answer_token_score_probs = all_score_probs[
                answer_start_token_idx - num_score_question_tokens
                :answer_end_token_idx - num_score_question_tokens
            ]
        answer_token_ids = sequence_ids[answer_start_token_idx:answer_end_token_idx].detach().cpu().long()

        return final_answer, answer_token_probs, answer_token_ids, answer_token_score_probs



    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Converts messages dict to a prompt_text with proper chat template applied"""
        has_assistant_prefill = any(m.get("role") == "assistant" for m in messages)

        if has_assistant_prefill:
            prompt_text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        else:
            prompt_text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,       # Let the template add the assistant header
                continue_final_message=False,
            )
        return prompt_text



    def process_generation_output(self, llm_outputs: LLMOutput) -> ParsedOutputGeneration:
        """
        Once we generate stuff from the transformer, we need to do a lot of parsing and processing to get it into the form we need for confidence scoring and evaluation. This function does all that.
        """
        outputs = llm_outputs.outputs
        offset_mappings = llm_outputs.offset_mappings

        # all_probs[k] = distribution that produced output_tokens[num_question_tokens + k],
        # where num_question_tokens = len(output_tokens) - all_probs.shape[0]
        # is computed downstream in _extract_answer_and_probs.
        #
        # outputs.logits comes in two different shapes depending on which API
        # produced it, and we have to normalize them to that semantics:
        #   generate path  -> tuple of T_new tensors each [1, vocab], one per
        #                     generated token (only the new tokens have logits).
        #   forward path   -> single tensor [1, T_delta, vocab] over the delta
        #                     tokens fed past the cache. logits[0, i] predicts
        #                     the token at delta position i+1.
        if isinstance(outputs.logits, tuple):
            # Generate path: each tuple element is already the distribution that
            # produced its generated token, so stacking gives the right semantics.
            all_probs: torch.Tensor = torch.stack([F.softmax(s, dim=-1).squeeze(0) for s in outputs.logits])
        else:
            # Forward path: drop the last row so row k = "produced delta token k+1"
            # i.e. output_tokens[cache_len + 1 + k]. The very first delta token's
            # producing logits live in the cached prefix and aren't returned by
            # model.__call__(), so we lose that one position -- this is fine here
            # because answer tokens always sit deep in the delta, not at its boundary.
            all_probs: torch.Tensor = F.softmax(outputs.logits[0, :-1, :], dim=-1)

        output_scores = getattr(outputs, "scores", None)
        if isinstance(output_scores, tuple):
            all_score_probs: torch.Tensor | None = torch.stack([F.softmax(s, dim=-1).squeeze(0) for s in output_scores])
        else:
            all_score_probs = None

        output_text= self.model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        output_tokens = self.model.tokenizer.convert_ids_to_tokens(outputs.sequences[0])

        start_assistant_text = "<|start_header_id|>assistant<|end_header_id|>"
        cot_start_idx = output_text.find(start_assistant_text) + len(start_assistant_text)

        answer_span: AnswerSpan | None = _locate_answer_span(self, output_text, search_start=cot_start_idx)

        # answer_span needed to get text_cot
        cot_steps, text_question, text_cot, text_cot_with_answer, whole_cache, question_cache = self._extract_cot(output_text, output_tokens, offset_mappings, outputs.past_key_values, outputs.sequences[0], cot_start_idx, answer_span)

        # answer_span needed to get final_answer and answer_token_probs
        final_answer, answer_token_probs, answer_token_ids, answer_token_score_probs = self._extract_answer_and_probs(output_text, output_tokens, offset_mappings, all_probs, all_score_probs, outputs.sequences[0], answer_span)

        return ParsedOutputGeneration(
            cot_steps=cot_steps,
            final_answer=final_answer,
            text_question=text_question,
            text_cot=text_cot,
            text_cot_with_answer=text_cot_with_answer,
            whole_cache=whole_cache,
            question_cache=question_cache,
            answer_token_probs=answer_token_probs,
            answer_token_ids=answer_token_ids,
            answer_token_score_probs=answer_token_score_probs,
        )



    def generate_helper(
        self,
        prompt: str,
        max_tokens: int,
        cache: Optional[Tuple],
        temperature: float
    ) -> LLMOutput:
        """
        2-phase generation; 
        1. generate the cot part, with stop strings LLAMA_STOP_STRINGS
        2. generate the answer part, with "The answer is \\boxed{"
        """
        # phase 1
        phase_1_outputs: LLMOutput = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            cache=cache,
            temperature=temperature,
            stop_strings=LLAMA_STOP_STRINGS
        )

        # phase 2
        phase_1_outputs_raw = phase_1_outputs.outputs
        phase_1_outputs_text = self.model.tokenizer.decode(
            phase_1_outputs_raw.sequences[0],
            skip_special_tokens=False,
        )

        phase_1_outputs_text, phase_2_cache = self._strip_trailing_special_token(
            phase_1_outputs_text,
            phase_1_outputs_raw.past_key_values,
        )
        phase_2_prompt = phase_1_outputs_text + "\nThe answer is \\boxed{"
        phase_2_outputs: LLMOutput = self.model.generate(
            prompt=phase_2_prompt,
            max_tokens=200,
            cache=phase_2_cache,
            temperature=temperature
        )

        # breakpoint()

        return phase_2_outputs
    


    def forward_pass_helper(
        self,
        prompt: str | list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> LLMOutput | ModelOutput:
        return self.model.forward(
            prompt=prompt,
            cache=cache,
            return_llm_output=return_llm_output,
        )


    def generate_batch_helper(
        self,
        prompt: str,
        max_tokens: int,
        cache: KVCache | None,
        temperature: float,
        num_sequences: int,
    ) -> list[LLMOutput]:
        """Batched 2-phase generation for N sequences from the same prompt.
        Phase 1: generate CoT with num_return_sequences=N and stop strings.
        Phase 2: for each sequence, append answer prefix and generate answer (batched as N different prompts).
        Returns N LLMOutput objects, one per sequence.
        """
        # Phase 1: generate N CoT sequences from the same prompt
        phase_1_raw = self.model.generate_multi_sequence(
            prompt=prompt,
            max_tokens=max_tokens,
            cache=cache,
            temperature=temperature,
            num_sequences=num_sequences,
            stop_strings=LLAMA_STOP_STRINGS,
        )
        # phase_1_raw.sequences: [N, seq_len], phase_1_raw.logits: tuple of [N, vocab] per new token

        N = num_sequences
        phase_2_prompts = []
        for i in range(N):
            seq_text = self.model.tokenizer.decode(
                phase_1_raw.sequences[i], skip_special_tokens=False
            )
            seq_text, _ = self._strip_trailing_special_token_text(seq_text)
            phase_2_prompts.append(seq_text + "\nThe answer is \\boxed{")

        # Phase 2: generate answer completions from N different prompts (batched)
        phase_2_raw = self.model.generate_batch(
            prompts=phase_2_prompts,
            max_tokens=200,
            temperature=temperature,
            stop_strings=None,
        )
        # phase_2_raw.sequences: [N, max_seq_len] (left-padded)

        # Unbatch into N LLMOutput objects
        results = []
        pad_id = self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id
        for i in range(N):
            seq_ids = phase_2_raw.sequences[i]
            # Strip left-padding
            non_pad = (seq_ids != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() > 0:
                start_idx = non_pad[0].item()
                seq_ids = seq_ids[start_idx:]
            else:
                seq_ids = seq_ids

            # Extract per-sequence logits from the batched tuple
            # phase_2_raw.logits is a tuple of [N, vocab] tensors, one per generated position
            per_seq_logits = tuple(
                logit_step[i:i+1, :] for logit_step in phase_2_raw.logits
            )
            per_seq_scores = tuple(
                score_step[i:i+1, :] for score_step in phase_2_raw.scores
            ) if hasattr(phase_2_raw, 'scores') and phase_2_raw.scores else None

            # Build a namespace that looks like single-sequence GenerateDecoderOnlyOutput
            per_seq_output = SimpleNamespace(
                sequences=seq_ids.unsqueeze(0),  # [1, seq_len]
                logits=per_seq_logits,
                scores=per_seq_scores,
                past_key_values=None,
            )

            output_text = self.model.tokenizer.decode(seq_ids, skip_special_tokens=False)
            full_offsets = self.model.tokenizer(
                output_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )["offset_mapping"]

            results.append(LLMOutput(outputs=per_seq_output, offset_mappings=full_offsets))

        return results


    def forward_pass_batch_helper(
        self,
        prompts: list[str],
        cache: KVCache | None = None,
        cache_seq_len: int = 0,
    ) -> list[LLMOutput]:
        """Batched forward pass over N different prompts.
        If cache is provided (and debug_nocache is off), uses cache-replicated batching.
        Otherwise falls back to full-prompt left-padded batching."""
        if cache is not None and not self.model.debug_nocache:
            # Tokenize each prompt, take tokens beyond cache_seq_len as delta
            delta_texts = []
            full_ids_list = []
            for p in prompts:
                ids = self.model.tokenizer(p, add_special_tokens=False).input_ids
                full_ids_list.append(ids)
                delta_ids = ids[cache_seq_len:]
                delta_texts.append(
                    self.model.tokenizer.decode(delta_ids, skip_special_tokens=False)
                )

            raw_outputs = self.model.forward_batch_with_cache(
                delta_texts=delta_texts,
                cache=cache,
                cache_seq_len=cache_seq_len,
                return_llm_output=True,
            )
            # Patch each output to have full sequences for process_generation_output
            results = []
            for i, raw in enumerate(raw_outputs):
                full_ids = torch.tensor(full_ids_list[i], dtype=torch.long).unsqueeze(0)
                # Prepend zero logits for prefix positions so indexing aligns with full sequence
                prefix_pad = torch.zeros(
                    1, cache_seq_len, raw.logits.shape[-1],
                    dtype=raw.logits.dtype, device=raw.logits.device
                )
                full_logits = torch.cat([prefix_pad, raw.logits], dim=1)  # [1, full_len, vocab]

                seq_output = SimpleNamespace(
                    logits=full_logits,
                    past_key_values=raw.past_key_values,
                    sequences=full_ids,
                )

                output_text = self.model.tokenizer.decode(full_ids[0], skip_special_tokens=False)
                full_offsets = self.model.tokenizer(
                    output_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )["offset_mapping"]

                results.append(LLMOutput(outputs=seq_output, offset_mappings=full_offsets))
            return results
        else:
            return self.model.forward_batch(prompts, return_llm_output=True)


    def _strip_trailing_special_token_text(self, text: str) -> tuple[str, int]:
        """Strip trailing special token from text, returning (cleaned_text, num_tokens_stripped)."""
        for special_tok in self.model.tokenizer.all_special_tokens:
            if text.endswith(special_tok):
                special_token_ids = self.model.tokenizer(
                    special_tok, add_special_tokens=False,
                ).input_ids
                return text[:-len(special_tok)], len(special_token_ids)
        return text, 0


