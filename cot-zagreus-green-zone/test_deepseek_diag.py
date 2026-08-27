"""Temp diagnostic: confirm whether continue_final_message works for deepseek-v4-flash-0731,
and whether the thinking-mode interaction is the real culprit.

Tests two scenarios for both Qwen (control) and Deepseek:
  A) Phase-2 boxing: assistant turn ending in '...\\boxed{', continue it, stop at '}'.
     Expect: the model emits the answer letter (e.g. 'C') as content.
  B) Forward-pass confidence: assistant turn ending in '...True/False:', continue it.
     Expect: the model emits 'True' or 'False' as the first content token.

For each scenario we toggle enable_thinking ON vs OFF and dump:
  - message.content, message.reasoning_content, finish_reason
  - the FULL logprob token stream (not just token 0) so we can see where
    <｜begin▁of▁sentence｜> / <|im_end|> / the answer token actually land.
"""

import os

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("no_proxy", ".huawei.com.*,10.0.0.0/8,localhost,127.0.0.1")

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

MODELS = ["Qwen3.8-27B", "deepseek-v4-flash-0731"]

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. For every problem you receive, think "
    "carefully and reason step-by-step. Label each reasoning step as 'Step 1:', "
    "'Step 2:', etc. Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY the single letter corresponding "
    "to the correct answer. Do not include periods, explanations, or any other text."
)

USER_CONTENT = (
    "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
    "Options:\n(A) The Front Page\n(B) Maelstrom\n(C) The Lion King\n(D) Lamerica\n\n"
    "Think carefully about the movies listed and the options provided. "
    "Reason step-by-step before choosing the best answer.\n\nLet's think step-by-step."
)

COT = (
    "\nStep 1: Batman, The Mask, The Fugitive, and Pretty Woman are all major "
    "mainstream Hollywood blockbusters from the early 1990s.\n\n"
    "Step 2: The Lion King (1994) is a major blockbuster from the same era, while "
    "the other options are not.\n\n"
    "Step 3: The Lion King fits the pattern."
)


def dump_logprob_stream(choice, max_show=25):
    lp = choice.logprobs
    if not lp or not lp.content:
        print("  (no logprobs)")
        return
    print(f"  logprob stream ({len(lp.content)} tokens):")
    for i, tlp in enumerate(lp.content[:max_show]):
        top3 = {t.token: round(t.logprob, 3) for t in tlp.top_logprobs[:3]}
        print(f"    [{i:2d}] {tlp.token!r:30} logprob={round(tlp.logprob, 4):>8}  top3={top3}")
    if len(lp.content) > max_show:
        print(f"    ... ({len(lp.content) - max_show} more)")


def run(label, model, messages, extra_body, max_tokens=20, stop=None):
    print(f"\n{'=' * 78}")
    print(f"  [{model}] {label}")
    print(f"{'=' * 78}")
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body=extra_body,
    )
    if stop:
        kwargs["stop"] = stop
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content
    reasoning = getattr(choice.message, "reasoning_content", None)
    print(f"  content:       {content!r}")
    print(f"  reasoning:     {reasoning!r}" if reasoning else "  reasoning:     (none)")
    print(f"  finish_reason: {choice.finish_reason}")
    dump_logprob_stream(choice)


for model in MODELS:
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_CONTENT},
    ]

    # ---- Scenario A: Phase-2 boxing (continue from '\boxed{') ----
    boxing_messages = base_messages + [
        {"role": "assistant", "content": COT + "\nThe answer is \\boxed{"}
    ]
    boxing_on = {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    boxing_off = {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    run("BOXING: continue from \\boxed{  | thinking ON", model,
        boxing_messages, boxing_on, max_tokens=16, stop=["}"])
    run("BOXING: continue from \\boxed{  | thinking OFF", model,
        boxing_messages, boxing_off, max_tokens=16, stop=["}"])

    # ---- Scenario B: Forward-pass confidence (continue from 'True/False:') ----
    fwd_messages = base_messages + [
        {"role": "assistant", "content": COT + "\nThe answer is \\boxed{C}.\nIs this answer True/False:"}
    ]
    fwd_on = {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    fwd_off = {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    run("FORWARD: continue from True/False:  | thinking ON", model,
        fwd_messages, fwd_on, max_tokens=20)
    run("FORWARD: continue from True/False:  | thinking OFF", model,
        fwd_messages, fwd_off, max_tokens=20)

    # ---- Scenario C: control — same boxing prompt but WITHOUT continue_final_message ----
    boxing_nocont = {
        "chat_template_kwargs": {"enable_thinking": True},
    }
    run("BOXING: NO continue_final_message  | thinking ON (control)", model,
        boxing_messages, boxing_nocont, max_tokens=16, stop=["}"])
