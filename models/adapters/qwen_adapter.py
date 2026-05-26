import re
import copy

from domain.data import LLMOutput, ParsedOutputGeneration, KVCache, CacheBundle, AnswerSpan


from models.adapters.base import ModelAdapter
from models.core_models.llm import LLM
from models.adapters.registry import MODEL_ATTENTION_IMPLEMENTATION_REGISTRY



import torch.nn.functional as F
import torch


class LlamaAdapter(ModelAdapter):
    def __init__(self):
        attention_implementation = MODEL_ATTENTION_IMPLEMENTATION_REGISTRY.get("llama")
        self.model = LLM(model_name="llama", attention_implementation=attention_implementation)

    def _locate_answer_span(self, output_text: str) -> AnswerSpan | None:
        m = re.search(r'\\boxed\{', output_text)
        if m is None:
            return None

        content_start = m.end()          # char after the opening '{'
        depth = 1
        i = content_start
        while i < len(output_text) and depth > 0:
            c = output_text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if depth != 0:
            return None  # Unmatched braces
        content_end = i - 1              # index of the closing '}'

        sentence_start = max(
            output_text.rfind('. ', 0, m.start()),
            output_text.rfind('\n', 0, m.start()),
        ) + 1
        while sentence_start < m.start() and output_text[sentence_start] == ' ':
            sentence_start += 1

        return AnswerSpan(
            char_answer_sentence_start=sentence_start,
            char_answer_boxed_start=content_start,
            char_answer_boxed_end=content_end,
        )

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


    def _char_to_token_idx(self, char_idx: int, offset_mappings: list[tuple[int, int]]) -> int:
        for token_idx, (start, end) in enumerate(offset_mappings):
            if start <= char_idx < end:
                return token_idx
        raise ValueError(f"Character index {char_idx} not found in any token span")

    def _extract_cot(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], cache: KVCache, sequence_ids: torch.Tensor, answer_span: AnswerSpan | None) -> tuple[list[str], str, str, CacheBundle, KVCache]:
        start_assistant_text = "<|start_header_id|>assistant<|end_header_id|>"
        cot_start_idx = output_text.find(start_assistant_text) + len(start_assistant_text)
        # text_cot_with_answer contains basically everything after the "assistant" header
        text_cot_with_answer = output_text[cot_start_idx:]

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



   
        if hasattr(cache, 'crop'):
            question_cache.crop(max(cot_start_token_idx - 1, 0))
            
        else:
            # Qwen3_5DynamicCache: crop only attention layers (non-None KV entries)
            # NOTE may be incorrect cropping here as Qwen3.5 also has conv_states and recurrent_states
            for idx in range(len(cache.key_cache)):
                if cache.key_cache[idx] is not None and cache.key_cache[idx].dim() == 4:
                    cache.key_cache[idx] = cache.key_cache[idx][:, :, :max_length, :]
                    cache.value_cache[idx] = cache.value_cache[idx][:, :, :max_length, :]


        whole_cache = CacheBundle(
            cache=cot_with_answer_cache,
            input_ids=sequence_ids[cot_start_token_idx:].detach().cpu().clone(),
        )

        return cot_steps, text_cot, text_cot_with_answer, whole_cache, question_cache



    # cot_steps: list[str]
    # whole_cache: KVCache
    # question_cache: KVCache







    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Converts messages dict to a prompt_text with proper chat template applied"""
        has_assistant_prefill = False
        if "assistant" in messages:
            has_assistant_prefill = True

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

        # See llama_adapter.process_generation_output for the full rationale.
        # outputs.logits arrives as a tuple of [1, vocab] (generate path) or as
        # a single [1, T_delta, vocab] (forward path); normalize to a 2D tensor
        # [T_scored, vocab] where row k is the distribution that produced the
        # k-th token after the question prefix.
        if isinstance(outputs.logits, tuple):
            all_probs: torch.Tensor = torch.stack([F.softmax(s, dim=-1).squeeze(0) for s in outputs.logits])
        else:
            all_probs: torch.Tensor = F.softmax(outputs.logits[0, :-1, :], dim=-1)

        output_text= self.model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        output_tokens = self.model.tokenizer.convert_ids_to_tokens(outputs.sequences[0])
        
        answer_span: AnswerSpan | None = self._locate_answer_span(output_text)

        cot_steps, text_cot, text_cot_with_answer, whole_cache, question_cache = self._extract_cot(output_text, output_tokens, offset_mappings, outputs.past_key_values, outputs.sequences[0], answer_span)

        final_answer, answer_token_probs = self._extract_answer_and_probs(output_text, output_tokens, offset_mappings, all_probs)

        return ParsedOutputGeneration(
            cot_steps=cot_steps,
            final_answer=final_answer,
            text_cot=text_cot,
            text_cot_with_answer=text_cot_with_answer,
            whole_cache=whole_cache,
            question_cache=question_cache,
            answer_token_probs=answer_token_probs,
        )







