import re
import copy
import logging

from typing import Optional, Tuple

from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob

from domain import LLMOutput, ParsedOutputGeneration, KVCache, CacheBundle, AnswerSpan, ScorerOutput

from models.adapters.base import ModelAdapter, ModelScorer
from models.core_models.api_llm import API_LLM
from models.adapters.registry import MODEL_ATTENTION_IMPLEMENTATION_REGISTRY, ANSWER_TOKENS
from models.adapters.shared_utils import _locate_answer_span, _char_to_token_idx



import torch.nn.functional as F
import torch


logger = logging.getLogger(__name__)




class GptScorer(ModelScorer):
    def __init__(self, model: API_LLM):
        self.model = model

    def forward_indirect(self, prompt: list[dict[str, str]], whole_cache: CacheBundle) -> ScorerOutput:
        """
        forward_indirect runs a forward pass on the prompt with indirect suffix, and returns the logits for the indirect tokens. These are 'True' and 'False' tokens generated last
        For GPT, whole_cache is not used.
        """
        llm_outputs: LLMOutput = self.model.forward(prompt)
        completion: ChatCompletion = llm_outputs.outputs
        first_token = completion.choices[0].logprobs.content[0]
        top_logprobs: TopLogprob = first_token.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}

        return {
            "True": torch.tensor(scores.get(ANSWER_TOKENS['gpt_True'][0], float("-inf"))),
            "False": torch.tensor(scores.get(ANSWER_TOKENS['gpt_False'][0], float("-inf"))),
        }

    def forward_verbal(self, prompt: list[dict[str, str]], whole_cache: CacheBundle) -> ScorerOutput:
        """
        forward_verbal runs a forward pass on the prompt with verbal suffix, and returns the logits for the verbal tokens. These are the tokens generated last.
        For llama, we are lucky such that every integer [0, 100] has its own token, so we only need one forward pass
        """

        llm_outputs: LLMOutput = self.model.forward(prompt)
        completion: ChatCompletion = llm_outputs.outputs
        first_token = completion.choices[0].logprobs.content[0]
        top_logprobs: TopLogprob = first_token.top_logprobs
        scores = {lp.token: lp.logprob for lp in top_logprobs}

        return {
            s: torch.tensor(scores.get(s, float("-inf")))
            for s in ANSWER_TOKENS['gpt_verbal_confidence']
        }






class GptAdapter(ModelAdapter):
    def __init__(self):
        self.model = API_LLM(model_name="gpt")
        self.model_scorer = GptScorer(self.model)

    def cost(self) -> float:
        return self.model.cost



    def _extract_cot(self, output_text: str, cot_start_idx: int, answer_span: AnswerSpan | None) -> tuple[list[str], str, str, str]:
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

        return cot_steps, text_question, text_cot, text_cot_with_answer


    def _extract_answer_and_probs(
        self, 
        output_text: str, 
        output_tokens: list[str], 
        offset_mappings: list[tuple[int, int]], 
        all_probs: list[TopLogprob], 
        answer_span: AnswerSpan | None
    ) -> tuple[str, list[TopLogprob], list[str]]:
        if answer_span is None:
            return "", [], []


        final_answer = output_text[answer_span.char_answer_boxed_start:answer_span.char_answer_boxed_end].strip()

        # these token indices are absolute
        answer_start_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_start, offset_mappings)
        answer_end_token_idx = _char_to_token_idx(self, answer_span.char_answer_boxed_end, offset_mappings)

        answer_token_probs = all_probs[answer_start_token_idx:answer_end_token_idx]
        answer_tokens = output_tokens[answer_start_token_idx:answer_end_token_idx]

        return final_answer, answer_token_probs, answer_tokens



    def _locate_answer_span_from_forward_pass(self, output_text: str) -> AnswerSpan | None:
        """
        output_text contains all of the generation of the "forward pass" tokens
        Usage right now is only for stepbootstrap sampling
        """
        # if the model outputs '}' , take everything before it
        end_idx = output_text.find('}')

        return AnswerSpan(
            char_answer_sentence_start=0,
            char_answer_boxed_start=0,
            char_answer_boxed_end=len(output_text) if end_idx == -1 else end_idx,
        )



    def render_prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """We don't do anything to the messages :P"""
        return messages



    def process_generation_output(self, llm_outputs: LLMOutput, type: str = "generate") -> ParsedOutputGeneration:
        """
        Once we generate stuff from the client, we need to do a lot of parsing and processing to get it into the form we need for confidence scoring and evaluation. This function does all that.
        Added an additional parameter to differentiate between generate and forward pass. Since forward pass for LLM_API is kinda tricky and need special handling.
        """
        outputs: ChatCompletion = llm_outputs.outputs
        choice: Choice = outputs.choices[0]
        output_text = choice.message.content

        log_probs: ChoiceLogprobs = choice.logprobs
        content: list[ChatCompletionTokenLogprob] = log_probs.content

        output_tokens: list[str] = []
        all_logprobs: list[float] = [] # stores the logprobs for all tokens generated
        all_top_logprobs: list[TopLogprob] = [] # stores the top logprobs for all tokens generated
        for token_logprob in content:
            output_tokens.append(token_logprob.token)
            all_logprobs.append(token_logprob.logprob)
            all_top_logprobs.append(token_logprob.top_logprobs)

        cot_start_idx = 0

        if type == "forward_pass":
            answer_span: AnswerSpan | None = self._locate_answer_span_from_forward_pass(output_text)
        else:
            answer_span: AnswerSpan | None = _locate_answer_span(self, output_text, search_start=cot_start_idx)

        # create the offset_mappings
        offset_mappings = []
        current_pos = 0
        for token in output_tokens:
            offset_mappings.append((current_pos, current_pos + len(token)))
            current_pos += len(token)

        # answer_span needed to get text_cot
        cot_steps, text_question, text_cot, text_cot_with_answer = self._extract_cot(
            output_text=output_text ,
            cot_start_idx=cot_start_idx,
            answer_span=answer_span
        )

        # answer_span needed to get final_answer and answer_token_probs
        final_answer, answer_token_probs, answer_tokens = self._extract_answer_and_probs(
            output_text=output_text,
            output_tokens=output_tokens,
            offset_mappings=offset_mappings,
            all_probs=all_top_logprobs,
            answer_span=answer_span
        )

        return ParsedOutputGeneration(
            cot_steps=cot_steps,
            final_answer=final_answer,
            text_question=text_question,
            text_cot=text_cot,
            text_cot_with_answer=text_cot_with_answer,
            whole_cache=None,
            question_cache=None,
            answer_token_probs=answer_token_probs,
            answer_token_ids=answer_tokens,
            input_messages=llm_outputs.input_messages,
        )



    def generate_helper(
        self,
        prompt: list[dict[str, str]],
        max_tokens: int,
        cache: Optional[Tuple],
        temperature: float
    ) -> LLMOutput:
        return self.model.generate(
            prompt_messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )


    def forward_pass_helper(
        self,
        prompt: list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
        return_llm_output: bool = False,
    ) -> LLMOutput:
        """
        the cache and return_llm_output are not used for API models
        """
        return self.model.forward(
            prompt_messages=prompt,
        )


    def forward_pass(
        self,
        messages: list[dict[str, str]],
        cache: Optional[CacheBundle] = None,
    ) -> ParsedOutputGeneration:
        """
        Overriding the base implementation to include param to differentiate between generate and forward pass.
        """
        prompt_text = self.render_prompt(messages)
        cache = self.align_cache(cache, prompt_text)
        output = self.forward_pass_helper(prompt_text, cache, return_llm_output=True)
        return self.process_generation_output(output, type="forward_pass")

