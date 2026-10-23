
from openai import OpenAI
from getpass import getpass
import time

client = OpenAI(
    base_url="http://127.0.0.1:18000/v1",
    api_key=getpass("API Key: "),
    timeout=180.0
)

model_name = client.models.list().data[0].id

start = time.perf_counter()

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are an expert in evolutionary game theory."
        },
        {
            "role": "user",
            "content": "What is the spatial Prisoner's Dilemma? Explain briefly."
        }
    ],
    temperature=0.7,
    max_tokens=512
)

elapsed = time.perf_counter() - start

print("模型：", model_name)
print("回答：", response.choices[0].message.content)
print("耗时：", round(elapsed, 2), "秒")
print("Token：", response.usage)

if response.usage:
    print(
        "平均输出速度：",
        round(
            response.usage.completion_tokens / elapsed, 2
        ),
        "tokens/s（包含首 Token 等待时间）"
    )
