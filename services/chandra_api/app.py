# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union
from openai import OpenAI
import base64
import uvicorn
import time 

app = FastAPI(title="Chandra Extract API")

# Configure the OpenAI-compatible client (local server)
client = OpenAI(base_url="http://localhost:7871/v1", api_key="not-needed")  # any string works
MODEL_NAME = "chandra"

class URLPayload(BaseModel):
    image_url: HttpUrl

def _data_url_from_upload(upload: UploadFile) -> str:
    # Infer MIME type from the uploaded file; default to jpeg if unknown
    mime = upload.content_type or "image/jpeg"
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

from PIL import Image  # <-- Required for format detection

def decode_image(data: Union[str, bytes]) -> str:
    """
    Smart decoder - handles both base64 strings and raw bytes.
    Detects the image format and returns a formatted data URL string
    like 'data:image/png;base64,...'
    """
    try:
        # --- 1. Decode input to raw bytes ---
        
        image_bytes: bytes
        
        # Case 1: It's a base64 string
        if isinstance(data, str):
            # Remove data URI prefix if present (e.g., "data:image/png;base64,")
            if ',' in data and data.startswith('data:'):
                data = data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(data)
        # Case 2: It's already bytes
        else:
            image_bytes = data
            
        if not image_bytes:
            raise ValueError("Input data is empty.")

        # --- 2. Detect MIME type from bytes ---
        
        mime_type = "image/png"  # Default, like in your original function
        try:
            # Open with PIL to find the format
            with Image.open(io.BytesIO(image_bytes)) as img:
                image_format = img.format
                if image_format:
                    # Look up the standard MIME type (e.g., 'JPEG' -> 'image/jpeg')
                    detected_mime = Image.MIME.get(image_format)
                    if detected_mime:
                        mime_type = detected_mime
        except Exception:
            # If PIL fails to open, it's either not an image or a format
            # PIL doesn't support. We'll stick with the default.
            pass

        # --- 3. Encode bytes to base64 string ---
        
        b64_string = base64.b64encode(image_bytes).decode("utf-8")

        # --- 4. Format as data URL ---
        
        return f"data:{mime_type};base64,{b64_string}"
        
    except Exception as e:
        raise ValueError(f"Failed to decode and format image: {str(e)}")
ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]
ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()


@app.post("/chandra/extract")
async def chandra_extract(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[str] = Form(None),
    body: Optional[URLPayload] = Body(default=None),
):
    start_time = time.time()
    try:
        if file is None and body is None and image is None:
            raise HTTPException(status_code=400, detail="Provide either a file upload or JSON with image_url.")
        if file is not None and body is not None and image is not None:
            raise HTTPException(status_code=400, detail="Provide only one: file OR image_url.")

        if file is not None:
            image_url = _data_url_from_upload(file)
        elif image is not None:
            image_url = decode_image(image)
        else:
            image_url = str(body.image_url)
        # print(image_url[:100])
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_ENDING},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            temperature=0.1
        )

        content = resp.choices[0].message.content if resp and resp.choices else ""
        processing_time = time.time() - start_time
        return JSONResponse(
            status_code=200,
            content={
                "model": MODEL_NAME,
                "result": content,
                "processing_time": processing_time
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        # Surface a concise error without leaking internals
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9670, reload=True)