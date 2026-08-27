import asyncio
import logging

import torch
from typing import Optional

from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

from domain import LLMOutput
from models.core_models.api_llm import API_LLM
from models.adapters.gpt_adapter import GptAdapter, GptScorer
from models.adapters.registry import ANSWER_TOKENS

logger = logging.getLogger(__name__)

# deepseek-v4-flash emits a reasoning prefix ('.', then this marker token) before
# the answer token when continuing an assistant turn with thinking enabled. The
# answer we want is the first content token AFTER this marker.
_THINK_MARKER = "uarda"

# System-prompt suffix that forward()/forward_pass() append to steer the model
# toward a single True/False answer. Duplicated from API_LLM.forward() because
# Deepseek needs thinking ON, which forward() doesn't support.
_TF_INSTRUCTION = (
    "You are also asked to evaluate your answer with True/False after. "
    "ONLY respond with a single 'True' or 'False'."
)


class DeepseekScorer(GptScorer):
    """Confidence scorer for deepseek-v4-flash.

    Deepseek is a reasoning model: when continuing an assistant turn with
    thinking enabled, the logprob stream is [reasoning tokens..., '<think_marker>',
    answer token, ...]. The inherited GptScorer reads logprobs.content[0], which
    is the reasoning token -- not the answer. We skip past the thinking prefix
    to find the real answer token.
    """

    def _deepseek_forward(self, prompt: list[dict[str, str]]) -> LLMOutput:
        """Forward pass with thinking ON.

        Deepseek only emits the answer token (True/False, a number, etc.) with
        thinking enabled. With thinking OFF it produces degenerate output ('.'
        instead of 'True'). API_LLM.forward() hardcodes thinking OFF, so we
        bypass it and call generate() directly.
        """
        prompt[0]["content"] += _TF_INSTRUCTION
        cont = bool(prompt and prompt[-1].get("role") == "assistant")
        return self.model.generate(
            prompt_messages=prompt,
            max_tokens=200,
            temperature=0.0,
            continue_final_message=cont,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    @staticmethod
    def _find_answer_token(
        llm_outputs: LLMOutput,
    ) -> Optional[ChatCompletionTokenLogprob]:
        """Return the logprob entry for the first content token after the thinking marker."""
        content = llm_outputs.outputs.choices[0].logprobs.content or []
        for i, tlp in enumerate(content):
            if _THINK_MARKER in tlp.token:
                if i + 1 < len(content):
                    return content[i + 1]
                break
        # Fallback: first token that isn't BOS/EOS
        for tlp in content:
            if "begin" not in tlp.token and "end" not in tlp.token.lower():
                return tlp
        return content[0] if content else None

    def forward_indirect(self, prompt, whole_cache):
        old = (
            "During your reasoning, do NOT reveal, hint at, or restate the final "
            "answer. Do not write lines like 'Answer:', 'Final answer:', any answer "
            "strings, or any concluding sentence. Stop immediately after your last "
            "numbered reasoning step."
        )
        prompt[0]["content"] = prompt[0]["content"].replace(old, "")

        llm_outputs = self._deepseek_forward(prompt)
        answer_tlp = self._find_answer_token(llm_outputs)
        if answer_tlp is None:
            return (
                {"True": torch.tensor(float("-inf")), "False": torch.tensor(float("-inf"))},
                {},
            )
        top_logprobs = answer_tlp.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}
        scorer_output = {
            "True": torch.tensor(scores.get(ANSWER_TOKENS["gpt_True"][0], float("-inf"))),
            "False": torch.tensor(scores.get(ANSWER_TOKENS["gpt_False"][0], float("-inf"))),
        }
        debug_info = {
            "first_token": answer_tlp.token,
            "top_logprobs": {lp.token: lp.logprob for lp in top_logprobs},
        }
        return scorer_output, debug_info

    def forward_verbal(self, prompt, whole_cache):
        llm_outputs = self._deepseek_forward(prompt)
        answer_tlp = self._find_answer_token(llm_outputs)
        if answer_tlp is None:
            return (
                {s: torch.tensor(float("-inf")) for s in ANSWER_TOKENS["gpt_verbal_confidence"]},
                {},
            )
        top_logprobs = answer_tlp.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}
        scorer_output = {
            s: torch.tensor(scores.get(s, float("-inf")))
            for s in ANSWER_TOKENS["gpt_verbal_confidence"]
        }
        debug_info = {
            "first_token": answer_tlp.token,
            "top_logprobs": {lp.token: lp.logprob for lp in top_logprobs},
        }
        return scorer_output, debug_info

    async def forward_indirect_async(self, prompt, whole_cache):
        old = (
            "During your reasoning, do NOT reveal, hint at, or restate the final "
            "answer. Do not write lines like 'Answer:', 'Final answer:', any answer "
            "strings, or any concluding sentence. Stop immediately after your last "
            "numbered reasoning step."
        )
        prompt[0]["content"] = prompt[0]["content"].replace(old, "")

        llm_outputs = await asyncio.to_thread(self._deepseek_forward, prompt)
        answer_tlp = self._find_answer_token(llm_outputs)
        if answer_tlp is None:
            return (
                {"True": torch.tensor(float("-inf")), "False": torch.tensor(float("-inf"))},
                {},
            )
        top_logprobs = answer_tlp.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}
        scorer_output = {
            "True": torch.tensor(scores.get(ANSWER_TOKENS["gpt_True"][0], float("-inf"))),
            "False": torch.tensor(scores.get(ANSWER_TOKENS["gpt_False"][0], float("-inf"))),
        }
        debug_info = {
            "first_token": answer_tlp.token,
            "top_logprobs": {lp.token: lp.logprob for lp in top_logprobs},
        }
        return scorer_output, debug_info

    async def forward_verbal_async(self, prompt, whole_cache):
        llm_outputs = await asyncio.to_thread(self._deepseek_forward, prompt)
        answer_tlp = self._find_answer_token(llm_outputs)
        if answer_tlp is None:
            return (
                {s: torch.tensor(float("-inf")) for s in ANSWER_TOKENS["gpt_verbal_confidence"]},
                {},
            )
        top_logprobs = answer_tlp.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}
        scorer_output = {
            s: torch.tensor(scores.get(s, float("-inf")))
            for s in ANSWER_TOKENS["gpt_verbal_confidence"]
        }
        debug_info = {
            "first_token": answer_tlp.token,
            "top_logprobs": {lp.token: lp.logprob for lp in top_logprobs},
        }
        return scorer_output, debug_info


class DeepseekAdapter(GptAdapter):
    """Adapter for deepseek-v4-flash, a reasoning model served via the Huawei Ascend gateway.

    Three differences from GptAdapter, all confirmed by diagnostic testing:

    1. Boxing: Deepseek immediately closes \\boxed{ with }, so phase 2 continues
       from 'The answer is ' (no box) instead. The box is wrapped synthetically
       in the merge step.

    2. Thinking: Deepseek only emits the answer token with enable_thinking=True.
       With thinking OFF it produces degenerate output ('.' instead of the
       answer). API_LLM.forward() hardcodes thinking OFF, so all forward-pass
       paths are overridden to call generate() with thinking ON.

    3. Reasoning prefix: The logprob stream is [reasoning, '<think_marker>',
       answer, ...]. The answer token is after the marker, not at index 0.
       The scorer skips the prefix; the forward-pass helper strips it from the
       logprob stream so process_generation_output sees aligned tokens.
    """

    def __init__(self):
        self.model = API_LLM(model_name="deepseek_flash")
        self.model_scorer = DeepseekScorer(self.model)

    def generate_helper(self, prompt, max_tokens, cache, temperature):
        # Phase 1: generate CoT with reasoning_effort.
        phase_1 = self.model.generate(
            prompt_messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort="medium",
        )
        message = phase_1.outputs.choices[0].message
        phase_1_text = getattr(message, "content", None) or ""
        phase_1_cot = self._strip_trailing_bare_answer(phase_1_text)

        # Phase 2: continue from 'The answer is ' (NOT \boxed{, which Deepseek
        # closes immediately with }). Thinking OFF so the answer token emits
        # directly into the content logprob stream without a reasoning prefix.
        prefix = "\nThe answer is "
        phase_2_messages = list(prompt) + [
            {"role": "assistant", "content": phase_1_cot + prefix}
        ]
        phase_2 = self.model.generate(
            prompt_messages=phase_2_messages,
            max_tokens=16,
            temperature=temperature,
            continue_final_message=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return self._merge_two_phase_deepseek(prompt, phase_1_cot, phase_2)

    def _merge_two_phase_deepseek(self, original_prompt, phase_1_cot, phase_2):
        """Merge phase-1 CoT with phase-2 answer into a single \\boxed{ANSWER} completion.

        Phase 2 runs with thinking OFF, so content logprobs are answer tokens
        directly (no reasoning prefix to skip). We prepend a synthetic token
        holding 'CoT...\\boxed{' and append a synthetic closing '}', preserving
        the real phase-2 content tokens and their top_logprobs between them so
        process_generation_output can locate the \\boxed{} span and read the
        answer's real logprobs.
        """
        completion = phase_2.outputs
        choice = completion.choices[0]
        content_logprobs = choice.logprobs.content or []

        box_prefix = "\nThe answer is \\boxed{"
        blob = phase_1_cot + box_prefix
        answer_text = choice.message.content or ""
        choice.message.content = blob + answer_text + "}"

        synthetic_prefix = ChatCompletionTokenLogprob(
            token=blob, bytes=None, logprob=0.0, top_logprobs=[]
        )
        synthetic_brace = ChatCompletionTokenLogprob(
            token="}", bytes=None, logprob=0.0, top_logprobs=[]
        )

        choice.logprobs.content = (
            [synthetic_prefix] + content_logprobs + [synthetic_brace]
        )

        return LLMOutput(
            outputs=completion,
            offset_mappings=None,
            text_question=self.model._render_messages(original_prompt),
            input_messages=original_prompt,
        )

    @staticmethod
    def _strip_reasoning_logprobs(llm_output: LLMOutput):
        """Remove reasoning tokens (up to and including the thinking marker) from the logprob stream.

        After stripping, the logprob stream contains only content tokens, which
        aligns with choice.message.content. This lets the inherited
        process_generation_output work unchanged.
        """
        choice = llm_output.outputs.choices[0]
        content_logprobs = choice.logprobs.content or []
        stripped = []
        past_marker = False
        for tlp in content_logprobs:
            if not past_marker:
                if _THINK_MARKER in tlp.token:
                    past_marker = True
                continue
            stripped.append(tlp)
        if stripped:
            choice.logprobs.content = stripped
        if choice.message.content is None and stripped:
            choice.message.content = "".join(tlp.token for tlp in stripped)

    def _deepseek_forward_pass(self, prompt: list[dict[str, str]]) -> LLMOutput:
        """Forward pass with thinking ON, then strip reasoning prefix from logprobs."""
        prompt[0]["content"] += _TF_INSTRUCTION
        cont = bool(prompt and prompt[-1].get("role") == "assistant")
        llm_output = self.model.generate(
            prompt_messages=prompt,
            max_tokens=200,
            temperature=0.0,
            continue_final_message=cont,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        self._strip_reasoning_logprobs(llm_output)
        return llm_output

    def forward_pass_helper(self, prompt, cache=None, return_llm_output=False):
        return self._deepseek_forward_pass(prompt)

    async def forward_pass_async(self, messages, cache=None):
        prompt_text = self.render_prompt(messages)
        cache = self.align_cache(cache, prompt_text)
        llm_output = await asyncio.to_thread(self._deepseek_forward_pass, prompt_text)
        return self.process_generation_output(llm_output, type="forward_pass")
