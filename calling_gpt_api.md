# Calling the GPT API

This note summarizes the GPT stepbootstrap call path. For each dataset datapoint, the current configuration creates 100 resampled chains of thought and sends them to GPT as 100 sequential Chat Completions requests.

## 1. Configure the run — `scripts/gpt/bm/gpt_bm.sh`, `main.py`, `config.py`

The BigBench Movie launcher selects the API backend, GPT adapter, and 100 stepbootstrap samples:

```bash
# scripts/gpt/bm/gpt_bm.sh
python3 main.py \
    --backend api \
    --dataset bigbench_movie \
    --model gpt \
    --nb_stepbootstrap_samples 100 \
    --max_tokens 1024 \
    --temperature 0.9
```

The CLI arguments are declared in `main.py` and converted into `GenerationConfig` and `SamplingConfig` in `config.py`.

## 2. Select the GPT adapter — `models/registry.py`, `pipeline/runner.py`

`models/registry.py` maps the CLI model name to its adapter:

```python
MODEL_ADAPTER_REGISTRY = MappingProxyType(
    {
        "llama": LlamaAdapter,
        "qwen": QwenAdapter,
        "qwen_vllm": QwenVllmAdapter,
        "gpt": GptAdapter,
    }
)
```

`pipeline/runner.py` constructs the selected adapter:

```python
adapter_cls = MODEL_ADAPTER_REGISTRY[self.generation_config.model]
self.model_adapter = adapter_cls()
```

## 3. Build the stepbootstrap prompts — `pipeline/sampling/stepbootstrap_sampling.py`

`pipeline/sampling/stepbootstrap_sampling.py` resamples the vanilla CoT steps with replacement. For the API backend, each resampled CoT is placed in an assistant message ending immediately before the answer:

```python
final_answer_sentence = "\nTherefore the final answer is \\boxed{"
assistant_message = {
    "role": "assistant",
    "content": alternative_cot + final_answer_sentence,
}
new_messages.append(assistant_message)
```

The intended next token is therefore the answer inside `\boxed{...}`.

## 4. Make 100 sequential forward-pass requests — `pipeline/sampling/stepbootstrap_sampling.py`

The same module loops over all stepbootstrap samples:

```python
for i in range(self.sampling_config.nb_stepbootstrap_samples):
    new_messages = self._add_assistant_message_to_messages(
        messages,
        alternative_cots[i],
    )

    generate_output = self.context.model_adapter.forward_pass(
        messages=new_messages,
        cache=self.context.reference_vanilla_question_cache,
    )

    generation_outputs.append(generate_output)
```

For GPT, this is not batched: 100 samples produce 100 serial API requests. Although a cache is passed into the adapter interface, the API implementation does not support KV-cache reuse.

## 5. Delegate through `GptAdapter` — `models/adapters/gpt_adapter.py`

`models/adapters/gpt_adapter.py` renders the messages, calls the core API wrapper, and parses the response as a forward-pass result:

```python
def forward_pass(self, messages, cache=None):
    prompt_text = self.render_prompt(messages)
    cache = self.align_cache(cache, prompt_text)
    output = self.forward_pass_helper(
        prompt_text,
        cache,
        return_llm_output=True,
    )
    return self.process_generation_output(output, type="forward_pass")
```

Its helper delegates to `API_LLM.forward()`:

```python
return self.model.forward(prompt_messages=prompt)
```

## 6. Send the Chat Completions request — `models/core_models/api_llm.py`, `models/core_models/registry.py`

`models/core_models/api_llm.py` creates the client from environment variables:

```python
self.client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

The actual stepbootstrap request is made by `API_LLM.forward()`:

```python
response = self.client.chat.completions.create(
    model=self.model_name,
    messages=prompt_messages,
    max_tokens=20,
    temperature=0.0,
    logprobs=True,
    top_logprobs=20,
    **kwargs,
)
```

When the prompt ends in an assistant message, the wrapper also asks compatible servers to continue that message:

```python
kwargs["extra_body"] = {
    "continue_final_message": True,
    "add_generation_prompt": False,
}
```

This method is a simulated forward pass: Chat Completions must generate token(s) to return their log probabilities. It also sends and prefills the full prompt on every request because `API_LLM.align_cache()` returns `None`.

The concrete model name and API pricing are defined in `models/core_models/registry.py`:

```python
MODEL_API_REGISTRY = MappingProxyType({
    "gpt": "gpt-4o-mini-2024-07-18",
})
```

## 7. Parse the answer and compute confidence — `models/adapters/gpt_adapter.py`, `confidence/confidence_engine.py`

`models/adapters/gpt_adapter.py` reads the completion text and token-level top logprobs into a `ParsedOutputGeneration` record.

After all 100 stepbootstrap answer requests finish, `confidence/confidence_engine.py` computes indirect and verbal confidence for every output. Each confidence method calls the GPT forward path again, so this adds two API requests per bootstrap:

```text
100 stepbootstrap answer requests
+ 100 indirect-confidence requests
+ 100 verbal-confidence requests
= 300 API requests per datapoint
```

## 8. Record timing — `pipeline/runner.py`

`pipeline/runner.py` measures the whole 100-request stepbootstrap generation loop once:

```python
T0 = time.perf_counter()
stepbootstrap_generation_outputs = self.stepbootstrap_sampling.generate()
T1 = time.perf_counter()
```

It then writes the same total into every stepbootstrap sample record:

```python
timings=Timings(
    generation_time=T1 - T0,
    confidence_time=stepbootstrap_confidence_times[i],
    total_confidence_time=T2 - T1,
)
```

Consequently, this calculation is the amortized mean generation time per bootstrap, not the measured latency of sample zero:

```python
mean_seconds_per_bootstrap = (
    stepbootstraps["generation_time"][id_][0]
    / len(stepbootstraps["generation_time"][id_])
)
```

## Relevant files

- `scripts/gpt/bm/gpt_bm.sh` — run configuration.
- `main.py` — CLI declarations.
- `config.py` — generation and sampling configuration objects.
- `models/registry.py` — model-to-adapter mapping.
- `pipeline/runner.py` — orchestration and aggregate timing.
- `pipeline/sampling/stepbootstrap_sampling.py` — prompt construction and sequential request loop.
- `models/adapters/gpt_adapter.py` — GPT-specific adapter and response parsing.
- `models/core_models/api_llm.py` — OpenAI client and actual Chat Completions call.
- `models/core_models/registry.py` — concrete GPT model name and pricing.
- `confidence/confidence_engine.py` — indirect and verbal confidence requests.
- `domain/data.py` — parsed output, confidence, and timing records.
- `repository/trajectory_repository.py` — trajectory JSON persistence.
