import re
import copy

from typing import Optional, Tuple

from transformers.utils import ModelOutput


from domain import LLMOutput, ParsedOutputGeneration, KVCache, CacheBundle, AnswerSpan, ScorerOutput
from models.adapters.base import ModelAdapter, ModelScorer
from models.core_models.llm import LLM
from models.adapters.registry import MODEL_ATTENTION_IMPLEMENTATION_REGISTRY, ANSWER_TOKENS
from models.adapters.shared_utils import _locate_answer_span, _char_to_token_idx


import torch.nn.functional as F
import torch




class QwenScorer(ModelScorer):
    def __init__(self, model: LLM):
        self.model = model

    def forward_indirect(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput:
        """
        forward_indirect runs a forward pass on the prompt with indirect suffix, and returns the logitsfor the indirect tokens. These are 'True' and 'False' tokens generated last
        """
        cache = self.model.align_cache(whole_cache, prompt)

        outputs = self.model.forward(prompt, cache=cache)
        last_logits = outputs.logits[0, -1, :]

        tok = self.model.tokenizer
        true_id = tok(ANSWER_TOKENS[' True'][0], add_special_tokens=False).input_ids[0]
        false_id = tok(ANSWER_TOKENS[' False'][0], add_special_tokens=False).input_ids[0]

        return {
            'True': last_logits[true_id].detach().cpu(),
            'False': last_logits[false_id].detach().cpu(),
        }


    def forward_verbal(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput:
        """
        forward_verbal runs a forward pass on the prompt with verbal suffix, and returns the logits for the verbal tokens. These are the tokens generated last.
        For llama, we are lucky such that every integer [0, 100] has its own token, so we only need one forward pass
        """
        cache = self.model.align_cache(whole_cache, prompt)

        outputs = self.model.forward(prompt, cache=cache)
        last_logits = outputs.logits[0, -1, :]

        tok = self.model.tokenizer
        return {
            s: last_logits[tok(s, add_special_tokens=False).input_ids[0]].detach().cpu()
            for s in ANSWER_TOKENS['llama_verbal_confidence']
        }


"""
Some remarks:
- qwen is always concidered to be in thinking mode
- cache slicing does not work for Qwen3_5DynamicCache. For generation on question_cache and whole_cache, we just run a forward pass.
"""
class QwenAdapter(ModelAdapter):
    def __init__(self):
        attention_implementation = MODEL_ATTENTION_IMPLEMENTATION_REGISTRY.get("qwen")
        self.model = LLM(model_name="qwen", attention_implementation=attention_implementation)
        self.model_scorer = QwenScorer(self.model)



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
                return by_blank
            cot_steps = [s.strip() for s in text_cot.splitlines() if s.strip()]


        # to get the question cache, we need to run a forward pass on the question part of the prompt

        question_outputs: LLMOutput = self.model.forward(
            prompt=text_question,
            return_llm_output=True,
        )
        question_cache = CacheBundle(
            cache=question_outputs.outputs.past_key_values[0],
            input_ids=question_outputs.outputs.input_ids[0],
        )
        whole_cache = CacheBundle(
            cache=copy.deepcopy(cache),
            input_ids=sequence_ids[:cache.get_seq_length()].detach().cpu().clone(),
        )

        return cot_steps, text_question, text_cot, text_cot_with_answer, whole_cache, question_cache


    def _extract_answer_and_probs(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], all_probs: torch.Tensor, all_score_probs: torch.Tensor | None, sequence_ids: torch.Tensor, answer_span: AnswerSpan | None) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if answer_span is None:
            return "", torch.tensor([]), torch.tensor([], dtype=torch.long), None

        final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()

        # these token indices are absolute
        answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
        answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)

        # all_probs is relative to the start of generation, so we need to shift these token indices by the number of question tokens
        num_question_tokens = len(output_tokens) - all_probs.shape[0]
        answer_token_probs = all_probs[answer_start_token_idx - num_question_tokens:answer_end_token_idx - num_question_tokens]
        if all_score_probs is None:
            answer_token_score_probs = None
        else:
            num_score_question_tokens = len(output_tokens) - all_score_probs.shape[0]
            answer_token_score_probs = all_score_probs[answer_start_token_idx - num_score_question_tokens:answer_end_token_idx - num_score_question_tokens]
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
        # avoid empty think blocks that might be auto-injected
        prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)
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

        start_assistant_text = "<|im_start|>assistant"
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
        2 phase generation
        - 1) generate the thinking part, with "</think>" as the stop string
        - 2) generate the answer part, with post-thinking prefill "let's think step by step. Step 1: "
        """

        # phase 1
        phase_1_outputs: LLMOutput = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            cache=cache,
            temperature=temperature,
            stop_strings=["</think>"]
        )

        # Decoding phase 1 outputs to get generated text from token IDs
        phase_1_outputs_raw = phase_1_outputs.outputs
        phase_1_generated_tokens = phase_1_outputs_raw.sequences[0]
        phase_1_outputs_text = self.model.tokenizer.decode(phase_1_generated_tokens, skip_special_tokens=False)

        # 2) generate the answer part, with assistant prefill "let's think step by step. Step 1: "
        phase_2_prompt = phase_1_outputs_text + "Let's think step by step. Step 1: "
        phase_2_outputs: LLMOutput = self.model.generate(
            prompt=phase_2_prompt,
            max_tokens=max_tokens,
            cache=phase_1_outputs_raw.past_key_values,
            temperature=temperature
        )

        return phase_2_outputs



    def forward_pass_helper(
        self,
        prompt: str,
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> LLMOutput | ModelOutput:
        return self.model.forward(
            prompt=prompt,
            cache=cache,
            return_llm_output=return_llm_output,
        )