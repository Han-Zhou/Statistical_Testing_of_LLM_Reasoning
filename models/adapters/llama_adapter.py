import re
import copy
import time

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



    def _extract_cot(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], cache: KVCache, sequence_ids: torch.Tensor, cot_start_idx: int, answer_span: AnswerSpan | None) -> tuple[list[str], str, str, str, KVCache, CacheBundle]:
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


