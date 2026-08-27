import logging
import os
from openai import AsyncOpenAI, OpenAI
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
        self.async_client = AsyncOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_name = MODEL_API_REGISTRY.get(model_name)
        self.cost = 0.0

    
    def generate(
        self,
        prompt_messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        continue_final_message: bool = False,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict] = None
    ) -> LLMOutput:
        logger.info(f"Generating text with model {self.model_name}")
        kwargs = {}
        if stop:
            kwargs["stop"] = stop
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        if extra_body is not None:
            kwargs["extra_body"] = dict(extra_body)

        if continue_final_message:
            # phase-2 of 2-phase generation: continue the assistant turn that ends
            # in '...\boxed{' instead of opening a new one, so the box is completed
            # in place. vLLM-style extension; stock OpenAI ignores it.
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "continue_final_message": True,
                "add_generation_prompt": False,

            }
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=20,
            **kwargs,
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

        If the prompt ends in an assistant message (the confidence methods append the
        '...True/False:' / '<confidence>' tail as an assistant turn), ask the server to
        continue that message instead of opening a new turn, so token 0's logprobs are
        the genuine next-token distribution after the tail. Requires a vLLM-style
        OpenAI-compatible server; stock OpenAI ignores the flag.
        """
        logger.info(f"(Fake) forward pass with model {self.model_name}")
        kwargs = {}
        if prompt_messages and prompt_messages[-1].get("role") == "assistant":
            kwargs["extra_body"] = {
                "continue_final_message": True,
                "add_generation_prompt": False,
                # Disable thinking for reasoning models (e.g. Qwen Ascend): the
                # confidence methods need the model to emit the answer token
                # (True/False, etc.) directly, not re-reason. Stock OpenAI ignores
                # this flag.
                "chat_template_kwargs": {"enable_thinking": False},
            }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_messages,
            max_tokens=200,   # not sure of how many answer tokens we need to generate
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            **kwargs,
        )

        me = response.choices[0].message
        res = getattr(me, "reasoning_content", None) or ""
        nr = getattr(me, "content", None) or ""

        # breakpoint()

        self._accumulate_cost(response)
        return LLMOutput(
            outputs=response,
            offset_mappings=None,
            text_question=self._render_messages(prompt_messages),
            input_messages=prompt_messages,
        )


    async def forward_async(
        self,
        prompt_messages: list[dict[str, str]],
    ) -> LLMOutput:
        """Async equivalent of ``forward`` for concurrent independent prompts."""
        logger.info(f"(Async fake) forward pass with model {self.model_name}")
        kwargs = {}
        if prompt_messages and prompt_messages[-1].get("role") == "assistant":
            kwargs["extra_body"] = {
                "continue_final_message": True,
                "add_generation_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=prompt_messages,
            max_tokens=20,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            **kwargs,
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

