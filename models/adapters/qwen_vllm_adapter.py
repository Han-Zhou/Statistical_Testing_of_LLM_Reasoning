import re
import copy
import math

from typing import Optional, Tuple

from transformers.utils import ModelOutput
from vllm import RequestOutput, SamplingParams
from vllm.outputs import CompletionOutput
from vllm.logprobs import Logprob

from domain import LLMOutput, ParsedOutputGeneration, KVCache, CacheBundle, AnswerSpan, ScorerOutput, ListAnswerTokenProbs
from models.adapters.base import ModelAdapter, ModelScorer
from models.core_models.vllm_llm import VLLM_LLM
from models.adapters.registry import  ANSWER_TOKENS
from models.adapters.shared_utils import _locate_answer_span, _char_to_token_idx


import torch.nn.functional as F
import torch


QWEN_STOP_STRINGS = [
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
QWEN_STOP_STRINGS.extend(f"\n\n{letter}\n" for letter in LETTERS)
QWEN_STOP_STRINGS.extend(f"\n\n({letter})\n" for letter in LETTERS)





class QwenVllmScorer(ModelScorer):
    def __init__(self, model: VLLM_LLM):
        self.model = model

    def _next_token_logprobs(self, prompt: str) -> dict[int, Logprob]:
        """Generate 1 token to get the next-token distribution after the prompt."""
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            skip_special_tokens=False,
        )
        outputs: list[RequestOutput] = self.model.model.generate([prompt], sampling)
        return outputs[0].outputs[0].logprobs[0]

    def forward_indirect(self, prompt: str, whole_cache: CacheBundle = None) -> tuple[ScorerOutput, dict]:
        """
        forward_indirect runs a forward pass on the prompt with indirect suffix, and returns the logits for the indirect tokens. These are 'True' and 'False' tokens generated last
        """
        next_logprobs = self._next_token_logprobs(prompt)

        tok = self.model.tokenizer
        true_id = tok(ANSWER_TOKENS[' True'][0], add_special_tokens=False).input_ids[0]
        false_id = tok(ANSWER_TOKENS[' False'][0], add_special_tokens=False).input_ids[0]

        true_lp = next_logprobs[true_id].logprob if true_id in next_logprobs else -100.0
        false_lp = next_logprobs[false_id].logprob if false_id in next_logprobs else -100.0

        return {
            'True': torch.tensor(true_lp),
            'False': torch.tensor(false_lp),
        }, {}


    def forward_verbal(self, prompt: str, whole_cache: CacheBundle = None) -> tuple[ScorerOutput, dict]:
        """
        forward_verbal runs a forward pass on the prompt with verbal suffix, and returns the logits for the verbal tokens. These are the tokens generated last.
        """
        next_logprobs = self._next_token_logprobs(prompt)

        tok = self.model.tokenizer
        result = {}
        for s in ANSWER_TOKENS['llama_verbal_confidence']:
            tid = tok(s, add_special_tokens=False).input_ids[0]
            lp = next_logprobs[tid].logprob if tid in next_logprobs else -100.0
            result[s] = torch.tensor(lp)
        return result, {}


"""
Some remarks:
- qwen is always concidered to be in thinking mode
- cache slicing does not work for Qwen3_5DynamicCache. For generation on question_cache and whole_cache, we just run a forward pass.
"""
class QwenVllmAdapter(ModelAdapter):
    def __init__(self):
        self.model = VLLM_LLM(model_name="qwen")
        self.model_scorer = QwenVllmScorer(self.model)


    def _extract_cot(
        self, 
        output_text: str, 
        cot_start_idx: int, 
        answer_span: AnswerSpan | None
    ) -> tuple[list[str], str, str, str]:
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

        return cot_steps, text_question, text_cot, text_cot_with_answer




    def _extract_answer_and_probs(
        self, 
        output_text: str, 
        output_tokens: list[str], 
        offset_mappings: list[tuple[int, int]], 
        all_probs: ListAnswerTokenProbs, 
        all_token_ids: list[int],
        answer_span: AnswerSpan | None
    ) -> tuple[str, ListAnswerTokenProbs, list[str]]:
        if answer_span is None:
            return "", [], []

        final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()

        answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
        answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)

        # # all_probs is relative to the start of generation, so we need to shift these token indices by the number of question tokens
        # num_question_tokens = len(output_tokens) - all_probs.shape[0]
        # answer_token_probs = all_probs[answer_start_token_idx - num_question_tokens:answer_end_token_idx - num_question_tokens]
        # answer_token_ids = sequence_ids[answer_start_token_idx:answer_end_token_idx].detach().cpu().long()

         # all_probs covers only generated tokens, so shift absolute indices by prompt length
        num_question_tokens = len(output_tokens) - len(all_probs)
        answer_token_probs = all_probs[answer_start_token_idx - num_question_tokens:answer_end_token_idx - num_question_tokens]
        answer_token_ids = [
            self.model.tokenizer.decode([tid])
            for tid in all_token_ids[answer_start_token_idx:answer_end_token_idx]
        ]

        return final_answer, answer_token_probs, answer_token_ids



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



    def process_generation_output(self, llm_outputs: list[LLMOutput]) -> list[ParsedOutputGeneration]:
        """
        Once we generate stuff from the transformer, we need to do a lot of parsing and processing to get it into the form we need for confidence scoring and evaluation. This function does all that.
        """
        parsed_outputs = []
        for llm_output in llm_outputs:
            outputs: RequestOutput = llm_output.outputs

            # since phase 2 & 3 always has n=1 in the sampling, we should always expect the length of CompletionOutputs to be 1
            assert(len(outputs.outputs) == 1)
            completion_output: CompletionOutput = outputs.outputs[0]

            all_raw_logprobs: list[dict[int, Logprob]] = completion_output.logprobs

            # shape of all_probs: [num_generated_tokens, retrieved_vocab_size=20]
            all_probs: ListAnswerTokenProbs = []
            
            for logprob_dict in all_raw_logprobs:
                probs = {token_id: math.exp(lp.logprob) for token_id, lp in logprob_dict.items()}
                all_probs.append(probs)


            co = completion_output
            colp = co.logprobs
            first = colp[0]

            # breakpoint()

            output_text = outputs.prompt + completion_output.text
            all_token_ids = list(outputs.prompt_token_ids) + list(completion_output.token_ids)
            # vLLM may include a trailing stop token (e.g. <|im_end|>) in token_ids
            # that isn't present in completion.text. Strip it so IDs align with text.
            special_ids = set(self.model.tokenizer.all_special_ids)
            while all_token_ids and all_token_ids[-1] in special_ids:
                all_token_ids.pop()
                if all_probs:
                    all_probs.pop()
            output_tokens = self.model.tokenizer.convert_ids_to_tokens(all_token_ids)

            # get the offset mappings by tokenizing the output text
            enc = self.model.tokenizer(
                output_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = enc["input_ids"]
            if ids != list(all_token_ids):
                actual_tokens = self.model.tokenizer.convert_ids_to_tokens(ids)
                raise ValueError(
                    "Retokenized output_text does not match vLLM token ids "
                    f"(lengths: retok={len(ids)}, vllm={len(all_token_ids)}): "
                    f"vllm_tokens: {output_tokens}\n"
                    f"actual_tokens: {actual_tokens}"
                )
            # the original offset_mappings is none; not used for vllm
            offset_mappings = enc["offset_mapping"]

            start_assistant_text = "<|im_start|>assistant"
            cot_start_idx = output_text.find(start_assistant_text) + len(start_assistant_text)

            answer_span: AnswerSpan | None = _locate_answer_span(self, output_text, search_start=cot_start_idx)

            # answer_span needed to get text_cot
            cot_steps, text_question, text_cot, text_cot_with_answer = self._extract_cot(
                output_text, 
                cot_start_idx, 
                answer_span
            )

            # answer_span needed to get final_answer and answer_token_probs
            final_answer, answer_token_probs, answer_token_ids = self._extract_answer_and_probs(
                output_text, 
                output_tokens, 
                offset_mappings, 
                all_probs, 
                all_token_ids,
                answer_span
            )

            parsed_outputs.append(
                ParsedOutputGeneration(
                    cot_steps=cot_steps,
                    final_answer=final_answer,
                    text_question=text_question,
                    text_cot=text_cot,
                    text_cot_with_answer=text_cot_with_answer,
                    whole_cache=None,
                    question_cache=None,
                    answer_token_probs=answer_token_probs,
                    answer_token_ids=answer_token_ids,
                )
            )

        return parsed_outputs


    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        cache: Optional[CacheBundle] = None,
        temperature: float = 0.0,
    ) -> ParsedOutputGeneration:
        """
        Override so that the signatures match (list[outputs] instead of outputs for process_generation_output)
        """
        prompt_text = self.render_prompt(messages)
        cleaned_texts, forward_outputs = self.generate_helper(prompt_text, max_tokens, cache, temperature)
        return self._process_forward_output(cleaned_texts[0], forward_outputs[0])


    def generate_helper(
        self,
        prompt: str,
        max_tokens: int,
        cache: Optional[Tuple],
        temperature: float,
        n: int = 1,
    ) -> tuple[list[str], list[RequestOutput]]:
        """
        3 phase generation
        - 1) generate the thinking part, with "</think>" as the stop string
        - 2) generate the cot part, with post-thinking prefill "let's think step by step. Step 1: "
        - 3) generate the answer part, with "The answer is \boxed{"
        """

        # phase 1
        # the returned list[LLMOutput] will be of length 1, because we only passed in one prompt
        phase_1_output: LLMOutput = self.model.generate(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop_strings=["</think>"],
            include_stop_str_in_output=False
        )[0]

        # vllm's RequestOutput exposes the input prompt as .prompt and the
        # generated continuation as .outputs[0].text (no .text on the request
        # itself). To chain phases, concat prompt + completion + suffix.
        # phase_1_llm_output contains all the sampled trajectories for this prompt
        phase_1_request_output: RequestOutput = phase_1_output.outputs


        phase_2_prompts = [
            phase_1_request_output.prompt + phase_1_request_output.outputs[i].text + "</think>\nLet's think step by step. \nStep 1: "
            for i in range(n)
        ]

        # phase_2_outputs will be of length n, because we passed in n prompts
        phase_2_outputs: list[LLMOutput] = self.model.generate(
            prompts=phase_2_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop_strings=QWEN_STOP_STRINGS,
        )

        phase_3_prompts = [
            phase_2_outputs[i].outputs.prompt + phase_2_outputs[i].outputs.outputs[0].text + "\nThe answer is \\boxed{"
            for i in range(n)
        ]

        # phase_3_outputs will be of length n, because we passed in n prompts
        phase_3_outputs: list[LLMOutput] = self.model.generate(
            prompts=phase_3_prompts,
            max_tokens=200,
            temperature=temperature,
            n=1,
        )

        # Build full text for each sample, strip thinking, re-forward for clean logprobs
        cleaned_texts = []
        for llm_output in phase_3_outputs:
            request_output: RequestOutput = llm_output.outputs
            completion = request_output.outputs[0]
            full_text = request_output.prompt + completion.text
            # Strip <think>...</think> blocks
            cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
            cleaned_texts.append(cleaned)

        # Forward pass on cleaned texts to get aligned logprobs (same as HF adapter)
        forward_outputs: list[RequestOutput] = self.model.forward(
            prompts=cleaned_texts,
            return_llm_output=False,
        )

        return cleaned_texts, forward_outputs



    def forward_pass_helper(
        self,
        prompt: str,
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> list[RequestOutput]:
        return self.model.forward(
            prompts=[prompt],
            return_llm_output=False,
        )

    def _process_forward_output(self, prompt_text: str, request_output: RequestOutput) -> ParsedOutputGeneration:
        """Process a forward-pass RequestOutput (max_tokens=0, prompt_logprobs populated)."""
        output_text = prompt_text
        all_token_ids = list(request_output.prompt_token_ids)
        output_tokens = self.model.tokenizer.convert_ids_to_tokens(all_token_ids)

        # prompt_logprobs[i] = distribution that predicted token i; [0] is None
        raw_prompt_logprobs = request_output.prompt_logprobs
        all_probs: ListAnswerTokenProbs = []
        for plp in raw_prompt_logprobs:
            if plp is None:
                all_probs.append({})
            else:
                all_probs.append({token_id: math.exp(lp.logprob) for token_id, lp in plp.items()})

        enc = self.model.tokenizer(
            output_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mappings = enc["offset_mapping"]

        start_assistant_text = "<|im_start|>assistant"
        cot_start_idx = output_text.find(start_assistant_text) + len(start_assistant_text)

        answer_span: AnswerSpan | None = _locate_answer_span(self, output_text, search_start=cot_start_idx)

        cot_steps, text_question, text_cot, text_cot_with_answer = self._extract_cot(
            output_text,
            cot_start_idx,
            answer_span
        )

        # For the forward path, all_probs covers the entire prompt (not just generated tokens)
        # so num_question_tokens = 0 effectively — indices are already absolute
        if answer_span is None:
            final_answer, answer_token_probs, answer_token_ids = "", [], []
        else:
            final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()
            answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
            answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)
            answer_token_probs = all_probs[answer_start_token_idx:answer_end_token_idx]
            answer_token_ids = [
                self.model.tokenizer.decode([tid])
                for tid in all_token_ids[answer_start_token_idx:answer_end_token_idx]
            ]

        return ParsedOutputGeneration(
            cot_steps=cot_steps,
            final_answer=final_answer,
            text_question=text_question,
            text_cot=text_cot,
            text_cot_with_answer=text_cot_with_answer,
            whole_cache=None,
            question_cache=None,
            answer_token_probs=answer_token_probs,
            answer_token_ids=answer_token_ids,
        )

    def forward_pass(
        self,
        messages: list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
    ) -> ParsedOutputGeneration:
        prompt_text = self.render_prompt(messages)
        request_outputs = self.forward_pass_helper(prompt_text, cache)
        return self._process_forward_output(prompt_text, request_outputs[0])