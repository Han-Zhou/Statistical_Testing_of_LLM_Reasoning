# QwenAscendAdapter Integration Plan

## Context

We want to use Qwen3.8-27B (served via Huawei's Ascend gateway, OpenAI-compatible API) as a model in the cot-zagreus pipeline. The smoke test (`test_qwen_ascend.py`) confirmed:

- Inference works through `API_LLM` with the existing OpenAI SDK plumbing.
- The model is a **reasoning model**: `message.reasoning_content` holds the CoT, `message.content` holds the final answer (`\n\n391`).
- Logprobs span the **entire generation** — reasoning tokens first, then `</think>` separator, then answer tokens — as a single contiguous stream. Reconstructing `"".join(token.token for token in logprobs.content)` yields `reasoning_content + </think> + content + <|im_end|>`.
- `forward()` with `continue_final_message=True` does **not** re-reason — the first generated token is the answer token, and `content` is `None`. This means the 2-phase generation and confidence-scoring forward passes will work.

The existing `GptAdapter` assumes a non-reasoning model where `message.content` holds everything. The fix is narrowly scoped: read `reasoning_content` in phase 1, then everything else follows the same merge/parse/score path.

## Approach: subclass GptAdapter

Create `QwenAscendAdapter(GptAdapter)` that overrides only `__init__` and `generate_helper`. The scorer, `process_generation_output`, `_merge_two_phase`, `forward_pass`, and async methods are all inherited unchanged.

### Why this works

The confidence methods (`confidence/verbal.py:36`, `confidence/indirect.py:31`) build their prompt as:
```
input_messages + [{"role": "assistant", "content": text_cot + tail}]
```
So `text_cot` must contain the CoT. In `process_generation_output`, `text_cot` is extracted from `output_text` (line 208), which is `choice.message.content`. The merge (`_merge_two_phase`) sets `choice.message.content = blob + answer_text + "}"` where `blob = phase_1_cot + prefix`. So if we ensure `phase_1_cot` contains the reasoning text, the merged `content` will hold `reasoning + \boxed{answer}`, and all downstream extraction (`_extract_cot`, `_locate_answer_span`, `_extract_answer_and_probs`) works correctly.

The scorer reads `logprobs.content[0].top_logprobs` from the `forward()` call. Since `forward()` with `continue_final_message=True` doesn't re-reason (confirmed by test: first token is the answer digit), the first logprob entry is the answer token distribution. No scorer changes needed.

## Files to modify

### 1. Create `models/adapters/qwen_ascend_adapter.py`

```python
class QwenAscendScorer(GptScorer):
    """Inherits all scoring logic from GptScorer unchanged.
    The Ascend gateway returns logprobs in the same OpenAI format,
    and forward() with continue_final_message=True does not re-reason,
    so logprobs.content[0] is the answer token distribution."""
    pass


class QwenAscendAdapter(GptAdapter):
    def __init__(self):
        self.model = API_LLM(model_name="qwen_ascend")
        self.model_scorer = QwenAscendScorer(self.model)

    def generate_helper(self, prompt, max_tokens, cache, temperature):
        # Phase 1: single generate() call. The model produces reasoning_content
        # (the CoT) and content (the answer) automatically. We take the CoT
        # from reasoning_content; content is discarded (phase 2 produces the
        # boxed answer with real logprobs).
        phase_1 = self.model.generate(
            prompt_messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        message = phase_1.outputs.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or ""
        phase_1_cot = self._strip_trailing_bare_answer(reasoning)

        # Phase 2: identical to GptAdapter — prefill \boxed{ and continue.
        prefix = "\nThe answer is \\boxed{"
        phase_2_messages = list(prompt) + [
            {"role": "assistant", "content": phase_1_cot + prefix}
        ]
        phase_2 = self.model.generate(
            prompt_messages=phase_2_messages,
            max_tokens=16,
            temperature=temperature,
            stop=["}"],
            continue_final_message=True,
        )
        return self._merge_two_phase(prompt, phase_1_cot, prefix, phase_2)
```

Key differences from `GptAdapter.generate_helper`:
- Phase 1 CoT comes from `message.reasoning_content` instead of `message.content`.
- No other changes — `_merge_two_phase`, `process_generation_output`, `_extract_cot`, `_extract_answer_and_probs`, `_locate_answer_span` all operate on the merged `choice.message.content` which now holds `reasoning + \boxed{answer}`.

### 2. `models/registry.py` — register the adapter

- Import `QwenAscendAdapter` from `models.adapters.qwen_ascend_adapter`
- Add `"qwen_ascend": QwenAscendAdapter` to `MODEL_ADAPTER_REGISTRY`

### 3. `main.py` — make it selectable

- Line ~143: add `"qwen_ascend"` to `--model` choices
- Lines 215-221: add `"qwen_ascend": "api"` to `required_backends`

### 4. `pipeline/runner.py` — enable async paths

- Lines 256, 267: change `model == "gpt"` to `model in ("gpt", "qwen_ascend")` so async API concurrency works
- (The `else` branch at line 53 handles the no-arg constructor, which is what `QwenAscendAdapter` uses)

### 5. `pipeline/sampling/stepbootstrap_sampling.py` — enable async

- Line 136: change `model != "gpt"` to `model not in ("gpt", "qwen_ascend")`

## Files NOT modified

- `models/core_models/registry.py` — `qwen_ascend` already in `MODEL_API_REGISTRY`, `Qwen3.8-27B` already in `MODEL_API_PRICING` (both done in earlier steps)
- `models/core_models/api_llm.py` — no changes needed; `generate()` and `forward()` work as-is
- `models/adapters/registry.py` (`ANSWER_TOKENS`) — the scorer reuses `gpt_True`, `gpt_False`, `gpt_verbal_confidence` keys (inherited from `GptScorer`)
- `confidence/` — confidence methods are model-agnostic; they use `parsed_output.text_cot` and `parsed_output.input_messages` which are populated correctly by the inherited `process_generation_output`
- `.env` — already configured with the gateway URL, API key, and `no_proxy`

## Verification

1. **Smoke test**: `python test_qwen_ascend.py` — confirms inference, reasoning_content, logprobs, and forward() all work (already passing).

2. **Adapter unit test**: extend `test_qwen_ascend.py` or create a quick check that:
   - Builds a `QwenAscendAdapter()`
   - Calls `adapter.generate(messages, max_tokens=512, cache=None, temperature=0.0)`
   - Verifies `ParsedOutputGeneration.cot_steps` is non-empty (reasoning was extracted)
   - Verifies `final_answer` is correct (e.g. "391" for 17*23)
   - Verifies `text_cot` contains the reasoning text
   - Verifies `answer_token_probs` is non-empty (answer tokens have real logprobs)

3. **Confidence scoring test**: call `adapter.scorer().forward_indirect(...)` and `forward_verbal(...)` with a confidence prompt to verify logprobs are returned correctly.

4. **End-to-end pipeline**: run `python main.py --model qwen_ascend --backend api --dataset <small_dataset> --sampling vanilla --confidence indirect` on a few datapoints to confirm the full generation + confidence flow works.
