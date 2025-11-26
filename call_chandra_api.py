from openai import OpenAI
import base64, os

client = OpenAI(base_url="http://localhost:9670/v1", api_key="not-needed")  # any string works

with open("/root/tungn197/IDP/outputs/curated_mwg2025_pages/pages/page_06/crop_elements/table/curated_mwg2025_006_0005.png", "rb") as f:
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
with open("result.hjtml", "w") as f:
    f.write(resp.choices[0].message.content)