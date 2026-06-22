# `models/adapters/llama_adapter.py`

## `class LlamaScorer(ModelScorer)`

fields
+ model: LLM

methods
+ __init__(self, model: LLM)
+ forward_indirect(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput
+ forward_verbal(self, prompt: str, whole_cache: CacheBundle) -> ScorerOutput
- forward_continuations(self, continuation_texts: list[str], cache: KVCache, last_prompt_token_id: int) -> tuple[torch.Tensor, torch.Tensor]

---

## `class LlamaAdapter(ModelAdapter)`

fields
+ model: LLM
+ model_scorer: LlamaScorer

methods
+ __init__(self)
- _slice_cache(self, cache: KVCache, start: int, end: int) -> KVCache
- _extract_cot(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], cache: KVCache, sequence_ids: torch.Tensor, cot_start_idx: int, answer_span: AnswerSpan | None) -> tuple[list[str], str, str, str, KVCache, CacheBundle]
- _extract_answer_and_probs(self, output_text: str, output_tokens: list[str], offset_mappings: list[tuple[int, int]], all_probs: torch.Tensor, answer_span: AnswerSpan | None) -> tuple[str, torch.Tensor]
+ align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> KVCache | None
+ render_prompt(self, messages: list[dict[str, str]]) -> str
+ process_generation_output(self, llm_outputs: LLMOutput) -> ParsedOutputGeneration
