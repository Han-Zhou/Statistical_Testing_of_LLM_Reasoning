import re
import copy
import math

from typing import Optional, Tuple

from transformers.utils import ModelOutput
from vllm import RequestOutput
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

    def forward_indirect(self, prompt: str) -> ScorerOutput:
        """
        forward_indirect runs a forward pass on the prompt with indirect suffix, and returns the logitsfor the indirect tokens. These are 'True' and 'False' tokens generated last
        """

        outputs = self.model.forward(prompt)
        last_logits = outputs.outputs[0].prompt_logprobs[-1]

        tok = self.model.tokenizer
        true_id = tok(ANSWER_TOKENS[' True'][0], add_special_tokens=False).input_ids[0]
        false_id = tok(ANSWER_TOKENS[' False'][0], add_special_tokens=False).input_ids[0]

        return {
            'True': last_logits[true_id].detach().cpu(),
            'False': last_logits[false_id].detach().cpu(),
        }


    def forward_verbal(self, prompt: str) -> ScorerOutput:
        """
        forward_verbal runs a forward pass on the prompt with verbal suffix, and returns the logits for the verbal tokens. These are the tokens generated last.
        For llama, we are lucky such that every integer [0, 100] has its own token, so we only need one forward pass
        """

        outputs = self.model.forward(prompt)
        last_logits = outputs.outputs[0].prompt_logprobs[-1]

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
        answer_span: AnswerSpan | None
    ) -> tuple[str, ]:
        if answer_span is None:
            return "", torch.tensor([]), torch.tensor([], dtype=torch.long), None

        final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()

        # these token indices are absolute
        answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
        answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)

        # all_probs is relative to the start of generation, so we need to shift these token indices by the number of question tokens
        num_question_tokens = len(output_tokens) - all_probs.shape[0]
        answer_token_probs = all_probs[answer_start_token_idx - num_question_tokens:answer_end_token_idx - num_question_tokens]
        answer_token_ids = sequence_ids[answer_start_token_idx:answer_end_token_idx].detach().cpu().long()

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
            all_token_ids = outputs.prompt_token_ids + completion_output.token_ids
            output_tokens = self.model.tokenizer.convert_ids_to_tokens(all_token_ids)

            # get the offset mappings by tokenizing the output text
            enc = self.model.tokenizer(
                output_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = enc["input_ids"]
            if (ids != all_token_ids[:-1]):
                actual_tokens = self.model.tokenizer.convert_ids_to_tokens(ids)
                raise ValueError(
                    "Retokenized output_text does not match vLLM token ids: "
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



    def generate_helper(
        self,
        prompt: str,
        max_tokens: int,
        cache: Optional[Tuple],
        temperature: float,
        n: int = 1,
    ) -> list[LLMOutput]:
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

        for request_output in phase_3_outputs:
            completion = request_output.outputs.outputs[0]
            completion.text = re.sub(
                r"<think>.*?</think>",
                "",
                completion.text,
                flags=re.DOTALL,
            ).strip()

        return phase_3_outputs



    def forward_pass_helper(
        self,
        prompt: str,
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> LLMOutput | ModelOutput:
        return self.model.forward(
            prompt=[prompt],
            return_llm_output=return_llm_output,
        )