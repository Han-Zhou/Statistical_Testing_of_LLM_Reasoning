import os

import httpx
from openai import OpenAI

# diable global proxy
# os.environ["no_proxy"] = "*.huawei.com.*,10.*"  # Windows environment 
os.environ["no_proxy"] = ".huawei.com.*,10.0.0.0/8,localhost,127.0.0.1"  # Linux environment


def call_llm(prompt: str, url: str, api_key: str, model: str, max_tokens: int = 1024) -> str:
    client = OpenAI(
        base_url=url,
        api_key=api_key,
        default_headers={"Content-Type": "application/json"},
        http_client=httpx.Client(verify=False),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    message = completion.choices[0].message
    answer = message.content.strip()
    cot = message.reasoning_content if hasattr(message, "reasoning_content") and message.reasoning_content else ""
    if cot:
        answer = f"<think>\n{cot.strip()}\n</think>\n\n{answer}"
    return answer

gateway_url = "http://llm-api.noah.huawei.com/v1"  # Gateway address
# api_key = "sk-IrPcHfxJTwZyCZB_wIvKAA"  # Replace with the API_KEY provided by the administrator
api_key = "sk-7hY3oYO43lZHTIrQgASr5Q"
model = "Qwen3.8-27B"  # Replace with the model name. Check available models at http://10.90.91.123:3000/

print(call_llm(
    prompt="1+1=?",
    url=gateway_url,
    api_key=api_key,
    model=model,
))

