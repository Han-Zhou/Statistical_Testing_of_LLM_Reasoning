import os

import httpx
from openai import OpenAI

os.environ["no_proxy"] = ".huawei.com.*,10.0.0.0/8,localhost,127.0.0.1"

gateway_url = "http://llm-api.noah.huawei.com/v1"
api_key = "sk-7hY3oYO43lZHTIrQgASr5Q"

client = OpenAI(
    base_url=gateway_url,
    api_key=api_key,
    default_headers={"Content-Type": "application/json"},
    http_client=httpx.Client(verify=False),
)

for m in sorted(client.models.list().data, key=lambda x: x.id):
    print(m.id)
