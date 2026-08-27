import logging
import copy
import os
import torch
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Optional, Tuple
from dotenv import load_dotenv

from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm import RequestOutput

from transformers import AutoTokenizer
# from transformers.utils import ModelOutput
# from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
# from transformers.cache_utils import DynamicCaache

from domain.data import LLMOutput, KVCache, CacheBundle
from models.core_models.registry import MODEL_HF_REGISTRY

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class _ServerLogprob:
    """Small vLLM-compatible wrapper for an OpenAI logprob value."""

    logprob: float


@dataclass
class _ServerCompletionOutput:
    text: str
    token_ids: list[int]
    logprobs: list[dict[int, _ServerLogprob]]


@dataclass
class _ServerRequestOutput:
    """Subset of vLLM's RequestOutput used by the Qwen adapter."""

    prompt: str
    prompt_token_ids: list[int]
    outputs: list[_ServerCompletionOutput]
    prompt_logprobs: list[dict[int, _ServerLogprob] | None] | None


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an OpenAI model or a decoded JSON mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_id(value: Any) -> int | None:
    """Parse vLLM's numeric or ``token_id:<id>`` JSON keys."""

    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("token_id:"):
        text = text[len("token_id:") :]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _logprob(value: Any) -> float:
    raw = _field(value, "logprob", value)
    return float(raw)


class VLLMServerClient:
    """OpenAI-compatible client exposing the subset of offline vLLM we use.

    vLLM's completion endpoint supports token IDs and prompt logprobs as
    request extensions. The response is normalized to RequestOutput-like
    objects so the existing Qwen adapter and parsing code can be reused.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ):
        normalized_url = base_url.rstrip("/")
        if not normalized_url.endswith("/v1"):
            normalized_url = f"{normalized_url}/v1"
        self.model_name = model_name
        self.base_url = normalized_url
        self.client = client or OpenAI(
            base_url=normalized_url,
            api_key=api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY",
        )

    def generate(self, prompts: list[str], sampling: SamplingParams) -> list[_ServerRequestOutput]:
        if not prompts:
            return []

        # These are vLLM-specific OpenAI completion extensions. In particular,
        # return_token_ids is needed to reconstruct the offline RequestOutput
        # shape used by Qwen's answer-token probability alignment.
        extra_body: dict[str, Any] = {
            "include_stop_str_in_output": bool(
                getattr(sampling, "include_stop_str_in_output", False)
            ),
            "skip_special_tokens": bool(
                getattr(sampling, "skip_special_tokens", True)
            ),
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        prompt_logprobs = getattr(sampling, "prompt_logprobs", None)
        if prompt_logprobs is not None:
            extra_body["prompt_logprobs"] = prompt_logprobs

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=getattr(sampling, "max_tokens", None),
            temperature=getattr(sampling, "temperature", None),
            stop=getattr(sampling, "stop", None),
            n=getattr(sampling, "n", 1),
            logprobs=getattr(sampling, "logprobs", None),
            extra_body=extra_body,
        )

        choices = list(getattr(response, "choices", []))
        num_outputs = int(getattr(sampling, "n", 1) or 1)
        expected_choices = len(prompts) * num_outputs
        if len(choices) != expected_choices:
            raise RuntimeError(
                "vLLM server returned an unexpected number of completion choices: "
                f"expected {expected_choices}, got {len(choices)}"
            )

        request_outputs: list[_ServerRequestOutput] = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_choices = choices[
                prompt_index * num_outputs : (prompt_index + 1) * num_outputs
            ]
            first_choice = prompt_choices[0]
            prompt_token_ids = self._prompt_token_ids(first_choice)
            prompt_logprobs_value = self._prompt_logprobs(first_choice)
            completion_outputs = [
                self._completion_output(choice) for choice in prompt_choices
            ]
            request_outputs.append(
                _ServerRequestOutput(
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=completion_outputs,
                    prompt_logprobs=prompt_logprobs_value,
                )
            )
        return request_outputs

    @staticmethod
    def _prompt_token_ids(choice: Any) -> list[int]:
        token_ids = _field(choice, "prompt_token_ids")
        if token_ids is None:
            raise RuntimeError(
                "vLLM server did not return prompt_token_ids; "
                "ensure return_token_ids is supported by the server"
            )
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _prompt_logprobs(
        choice: Any,
    ) -> list[dict[int, _ServerLogprob] | None] | None:
        raw_prompt_logprobs = _field(choice, "prompt_logprobs")
        if raw_prompt_logprobs is None:
            return None
        converted: list[dict[int, _ServerLogprob] | None] = []
        for position in raw_prompt_logprobs:
            if position is None:
                converted.append(None)
                continue
            converted_position: dict[int, _ServerLogprob] = {}
            for raw_token_id, raw_value in position.items():
                token_id = _token_id(raw_token_id)
                if token_id is not None:
                    converted_position[token_id] = _ServerLogprob(_logprob(raw_value))
            converted.append(converted_position)
        return converted

    @classmethod
    def _completion_output(cls, choice: Any) -> _ServerCompletionOutput:
        token_ids_value = _field(choice, "token_ids")
        if token_ids_value is None:
            raise RuntimeError(
                "vLLM server did not return token_ids; "
                "ensure return_token_ids is supported by the server"
            )
        token_ids = [int(token_id) for token_id in token_ids_value]
        raw_logprobs = _field(choice, "logprobs")
        return _ServerCompletionOutput(
            text=str(_field(choice, "text", "")),
            token_ids=token_ids,
            logprobs=cls._completion_logprobs(raw_logprobs, token_ids),
        )

    @staticmethod
    def _completion_logprobs(
        raw_logprobs: Any,
        token_ids: list[int],
    ) -> list[dict[int, _ServerLogprob]]:
        if raw_logprobs is None:
            return []
        top_logprobs = _field(raw_logprobs, "top_logprobs", []) or []
        token_logprobs = _field(raw_logprobs, "token_logprobs", []) or []
        converted: list[dict[int, _ServerLogprob]] = []
        for index, raw_top in enumerate(top_logprobs):
            position: dict[int, _ServerLogprob] = {}
            for raw_token_id, raw_value in (raw_top or {}).items():
                token_id = _token_id(raw_token_id)
                if token_id is not None:
                    position[token_id] = _ServerLogprob(_logprob(raw_value))
            # The selected token is normally included in top_logprobs, but add
            # it explicitly for servers/configurations that omit it.
            if index < len(token_ids) and index < len(token_logprobs):
                position.setdefault(
                    token_ids[index], _ServerLogprob(_logprob(token_logprobs[index]))
                )
            converted.append(position)
        return converted


class VLLM_LLM():

    def __init__(self, model_name: str, base_url: str | None = None):
        self.model_name = model_name
        self.base_url = base_url
        # self.attention_implementation = attention_implementation
        self._load_model()


    def _load_model(self):
        # Load the model based on the model name
        logger.info(f"Loading VLLM_LLM model: {self.model_name}")
        actual_model_name = MODEL_HF_REGISTRY.get(self.model_name)
        if not actual_model_name:
            raise ValueError(f"Model {self.model_name} not found in registry.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name)

        if self.base_url is not None:
            logger.info(
                "Using remote vLLM server at %s; skipping local engine initialization",
                self.base_url,
            )
            self.model = VLLMServerClient(
                base_url=self.base_url,
                model_name=actual_model_name,
            )
            return

        hf_overrides =None
        if self.model_name in {"qwen", "qwen_vllm"}:
            hf_overrides = {"architectures": ["Qwen3_5ForCausalLM"]}

        self.model = LLM(
            model=actual_model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=torch.bfloat16,
            quantization="fp8",
            hf_overrides=hf_overrides,
        )
        
        logger.info(f"Model {self.model_name} loaded successfully.")


    def generate(
            self, 
            prompts: list[str], 
            max_tokens: int, 
            temperature: float,
            n: int,
            stop_strings: list[str] | None = None,
            include_stop_str_in_output: bool = False,
        ) -> list[LLMOutput]:
        # Generate text based on the prompt
        # prompt SHOULD ALREADY HAVE chat template applied
        logger.info(f"Generating text with model {self.model_name}")

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_strings,
            include_stop_str_in_output=include_stop_str_in_output,
            logprobs=20,
            skip_special_tokens=False,
            n=n,
        )

        outputs: list[RequestOutput] = self.model.generate(prompts, sampling)
        
        return [LLMOutput(outputs=output, offset_mappings=None) for output in outputs]
    


    def forward(
        self,
        prompts: list[str],
        return_llm_output: bool = False,
    ) -> list[RequestOutput] | list[LLMOutput]:
        """
        Single forward pass.
        - return_llm_output=False: returns the raw RequestOutput (used for confidence scoring).
        - return_llm_output=True: returns the LLMOutput
        """
        logger.info(f"Forward pass with model {self.model_name}")

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=20,
            skip_special_tokens=False,
        )

        outputs: list[RequestOutput] = self.model.generate(prompts, sampling)

        if not return_llm_output:
            return outputs

        return [LLMOutput(outputs=output, offset_mappings=None) for output in outputs]


    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> None:
        """
        we don't support caching for vllm
        """
        return None
