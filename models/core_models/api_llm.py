import logging
import os
from openai import OpenAI
from typing import Optional

from openai.types.chat.chat_completion import ChatCompletion

from models.core_models.registry import MODEL_API_REGISTRY, MODEL_API_PRICING
from domain import LLMOutput, KVCache, CacheBundle

logger = logging.getLogger(__name__)


class API_LLM():
    def __init__(self, model_name: str):
        # this is not a good design pattern, but right now only one API model is supported
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_name = MODEL_API_REGISTRY.get(model_name)
        self.cost = 0.0

    
    def generate(
        self,
        prompt_messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0
    ) -> LLMOutput:
        logger.info(f"Generating text with model {self.model_name}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=20,
        )
        self._accumulate_cost(response)
        return LLMOutput(
            outputs=response,
            offset_mappings=None,
            text_question=self._render_messages(prompt_messages),
            input_messages=prompt_messages,
        )


    def forward(
        self,
        prompt_messages: list[dict[str, str]],
    ) -> LLMOutput:
        """
        Technically not a forward pass
        Use case: Indirect, Verbal, and Stepbootstrap (WITHOUT FINAL ANSWER)confidence scoring
        So we generate a limited number of tokens to get the logits for the confidence scoring
        """
        logger.info(f"(Fake) forward pass with model {self.model_name}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_messages,
            max_tokens=20,   # not sure of how many answer tokens we need to generate
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        self._accumulate_cost(response)
        return LLMOutput(
            outputs=response,
            offset_mappings=None,
            text_question=self._render_messages(prompt_messages),
            input_messages=prompt_messages,
        )


    def _accumulate_cost(self, response: ChatCompletion) -> None:
        price = MODEL_API_PRICING[self.model_name]
        usage = response.usage
        self.cost += (
            usage.prompt_tokens * price["input"]
            + usage.completion_tokens * price["output"]
        )
        logger.info(f"Cumulative API cost: ${self.cost:.4f}")


    @staticmethod
    def _render_messages(messages: list[dict[str, str]]) -> str:
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"





    # align_cache always returns None; this does not support caching at all
    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> Optional[KVCache]:
        return None


