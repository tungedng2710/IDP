from openai import OpenAI
import base64, os

client = OpenAI(base_url="http://localhost:7871/v1", api_key="not-needed")  # any string works

with open("test_table.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

resp = client.chat.completions.create(
    model="chandra",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]
    }],
    temperature=0.1
)
print(resp.choices[0].message.content)