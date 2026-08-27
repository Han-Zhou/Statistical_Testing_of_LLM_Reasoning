"""
Smoke test for the Ascend Qwen3.8-27B endpoint via API_LLM.

Checks:
  1) Can we do inference at all (a basic chat completion)?
  2) What information comes back? In particular:
     - message.content (the answer text)
     - message.reasoning_content (the CoT, if the model emits it)
     - choice.logprobs (token logprobs / top_logprobs)
     - response.usage (token counts)

Run from the repo root:
    OPENAI_BASE_URL=http://llm-api.noah.huawei.com/v1 \
    OPENAI_API_KEY=sk-7hY3oYO43lZHTIrQgASr5Q \
    python test_qwen_ascend.py
"""

import os

from dotenv import load_dotenv
load_dotenv()

# Match ascend_llm_example.py: bypass the corporate proxy for the Huawei gateway.
os.environ.setdefault("no_proxy", ".huawei.com.*,10.0.0.0/8,localhost,127.0.0.1")

from models.core_models.api_llm import API_LLM


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # API_LLM reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment.
    if not os.getenv("OPENAI_BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_BASE_URL and OPENAI_API_KEY first.\n"
            "Example:\n"
            "  OPENAI_BASE_URL=http://llm-api.noah.huawei.com/v1 \\\n"
            "  OPENAI_API_KEY=sk-7hY3oYO43lZHTIrQgASr5Q \\\n"
            "  python test_qwen_ascend.py"
        )

    section("1) Building API_LLM(model_name='qwen_ascend')")
    model = API_LLM(model_name="qwen_ascend")
    print(f"model_name resolved to: {model.model_name}")
    print(f"base_url:               {os.getenv('OPENAI_BASE_URL')}")

    section("2) Inference via API_LLM.generate()")
    # generate() requests logprobs=True, top_logprobs=20 just like the real path.
    messages = [{"role": "user", "content": "What is 17 multiplied by 23? Answer with the number only."}]
    try:
        llm_output = model.generate(
            prompt_messages=messages,
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as e:
        print(f"generate() FAILED: {type(e).__name__}: {e}")
        return

    completion = llm_output.outputs
    choice = completion.choices[0]
    message = choice.message

    breakpoint()

    section("3) What came back?")
    print(f"finish_reason: {choice.finish_reason}")

    print("\n--- message.content ---")
    print(repr(message.content))

    print("\n--- message.reasoning_content ---")
    reasoning = getattr(message, "reasoning_content", None)
    print(f"has attribute: {hasattr(message, 'reasoning_content')}")
    if reasoning:
        preview = reasoning if len(reasoning) <= 800 else reasoning[:800] + f"\n... [truncated, {len(reasoning)} chars total]"
        print(preview)
    else:
        print("(empty or absent)")

    print("\n--- logprobs ---")
    lp = choice.logprobs
    print(f"logprobs object present: {lp is not None}")
    if lp is not None:
        content_lp = lp.content
        print(f"logprobs.content present: {content_lp is not None}")
        if content_lp:
            print(f"num logprob entries: {len(content_lp)}")
            print("\nfirst 5 token logprobs:")
            for i, tlp in enumerate(content_lp[:5]):
                top = {lp.token: round(lp.logprob, 4) for lp in tlp.top_logprobs}
                print(f"  [{i}] token={tlp.token!r:20} logprob={round(tlp.logprob, 4):>8}  top_logprobs={top}")
            print("\nlast 8 token logprobs:")
            for i, tlp in enumerate(content_lp[-8:], start=len(content_lp) - 8):
                top = {lp.token: round(lp.logprob, 4) for lp in tlp.top_logprobs}
                print(f"  [{i}] token={tlp.token!r:20} logprob={round(tlp.logprob, 4):>8}  top_logprobs={top}")

            # Reconstruct the full text from logprob tokens and compare against
            # reasoning_content + content to see if logprobs span both.
            reconstructed = "".join(tlp.token for tlp in content_lp)
            print("\n--- reconstructed from logprob tokens ---")
            print(repr(reconstructed))
            full_expected = (reasoning or "") + (message.content or "")
            print(f"\nreasoning_content + content = {repr(full_expected)}")
            print(f"reconstructed == reasoning+content: {reconstructed == full_expected}")
            print(f"reconstructed endswith content:      {reconstructed.endswith(message.content or '')}")
        else:
            print("logprobs.content is empty (logprobs not returned by the server)")
    else:
        print("logprobs is None (server did not return logprobs)")

    print("\n--- usage ---")
    if completion.usage:
        print(f"prompt_tokens:     {completion.usage.prompt_tokens}")
        print(f"completion_tokens: {completion.usage.completion_tokens}")
        print(f"total_tokens:      {completion.usage.total_tokens}")
    else:
        print("usage is None")

    section("4) forward() path (the confidence-scoring fake forward)")
    # forward() is what GptScorer relies on: it appends an assistant tail and
    # asks the server to continue it, then reads logprobs.content[0].top_logprobs.
    fwd_messages = messages + [
        {"role": "assistant", "content": "The answer is \\boxed{"},
    ]
    try:
        fwd_output = model.forward(prompt_messages=fwd_messages)
    except Exception as e:
        print(f"forward() FAILED: {type(e).__name__}: {e}")
        return

    fwd_choice = fwd_output.outputs.choices[0]
    print(f"finish_reason: {fwd_choice.finish_reason}")
    print(f"content:       {fwd_choice.message.content!r}")

    fwd_lp = fwd_choice.logprobs
    if fwd_lp is not None and fwd_lp.content:
        first = fwd_lp.content[0]
        top = {lp.token: round(lp.logprob, 4) for lp in first.top_logprobs}
        print(f"first token logprob: token={first.token!r} logprob={round(first.logprob, 4)}")
        print(f"first token top_logprobs ({len(first.top_logprobs)} entries): {top}")
        print("\n>>> forward() can supply top_logprobs: YES")
    else:
        print(">>> forward() can supply top_logprobs: NO (logprobs missing)")

    section("Summary")
    print(f"inference works:        YES")
    print(f"content present:        {bool(message.content)}")
    print(f"reasoning_content:      {bool(reasoning)}")
    print(f"logprobs present:       {lp is not None and bool(lp.content)}")
    print(f"top_logprobs (forward): {fwd_lp is not None and bool(fwd_lp.content) and bool(fwd_lp.content[0].top_logprobs)}")


if __name__ == "__main__":
    main()
