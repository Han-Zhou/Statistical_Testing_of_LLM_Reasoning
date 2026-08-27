"""Diagnostic: replicate the exact indirect-confidence forward pass on Ascend.

Tests whether the model continues the assistant's 'True/False:' suffix
or re-answers the question, with and without continue_final_message.
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

SYSTEM_PROMPT = """\
You are an expert reasoning assistant. For every problem you receive, think \
carefully and reason step-by-step. Label each reasoning step as 'Step 1:', \
'Step 2:', etc. Keep your reasoning concise, and be brief in each step.

When you output the final answer, output ONLY the single letter corresponding \
to the correct answer. Do not include periods, explanations, or any other text.
"""

COT = """
Step 1: Batman, The Mask, The Fugitive, and Pretty Woman are all major mainstream Hollywood blockbusters from the early 1990s.

Step 2: The Lion King (1994) is a major blockbuster from the same era, while the other options are not.

Step 3: The Lion King fits the pattern.

The answer is \\boxed{C}.
Is this answer True/False:"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT.strip()},
    {"role": "user", "content": (
        "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
        "Options:\n(A) The Front Page\n(B) Maelstrom\n(C) The Lion King\n(D) Lamerica\n\n"
        "Think carefully about the movies listed and the options provided. "
        "Reason step-by-step before choosing the best answer.\n\nLet's think step-by-step."
    )},
    {"role": "assistant", "content": COT},
]


def run(label: str, extra_body: dict | None, model: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  [{model}] {label}")
    print(f"{'=' * 70}")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body=extra_body or {},
    )
    choice = resp.choices[0]
    content = choice.message.content
    reasoning = getattr(choice.message, "reasoning_content", None)
    print(f"content:        {content!r}")
    print(f"reasoning:      {reasoning!r}")
    print(f"finish_reason:  {choice.finish_reason}")
    if choice.logprobs and choice.logprobs.content:
        first = choice.logprobs.content[0]
        print(f"first token:    {first.token!r}  (logprob={first.logprob:.4f})")
        top = {lp.token: round(lp.logprob, 4) for lp in first.top_logprobs[:10]}
        print(f"top-10 logprobs: {top}")
    else:
        print("no logprobs")


for model in MODELS:
    # 1) With continue_final_message + thinking off (current forward() behavior)
    run("WITH continue_final_message + enable_thinking=False", {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, model)

    # 2) Without continue_final_message, thinking off (new turn)
    run("WITHOUT continue_final_message + enable_thinking=False", {
        "chat_template_kwargs": {"enable_thinking": False},
    }, model)

    # 3) With continue_final_message + thinking ON
    run("WITH continue_final_message + enable_thinking=True", {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }, model)
