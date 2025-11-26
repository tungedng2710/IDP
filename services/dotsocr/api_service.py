from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import os
import sys
import logging
import base64
import json
from typing import Optional, List

# Add dots.ocr to path
sys.path.insert(0, '/app/dots.ocr')

from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="dots.ocr API Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VLLM_IP = os.getenv("VLLM_IP", "vllm-dots-ocr")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
MODEL_NAME = os.getenv("MODEL_NAME", "rednote-hilab/dots.ocr")


# Helper functions
def decode_image(data):
    """
    Smart decoder - handles both base64 strings and raw bytes
    Returns PIL Image
    """
    try:
        # Case 1: It's a base64 string
        if isinstance(data, str):
            # Remove data URI prefix if present
            if ',' in data and data.startswith('data:'):
                data = data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(data)
        # Case 2: It's already bytes (from file upload)
        else:
            image_bytes = data
        
        # Open with PIL
        image = Image.open(io.BytesIO(image_bytes))
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "dots.ocr API",
        "vllm_endpoint": f"http://{VLLM_IP}:{VLLM_PORT}",
        "endpoints": {
            "/get-text": "OCR text extraction (prompt_ocr)",
            "/extract": "Full layout extraction (prompt_layout_all_en)",
            "/get-layout": "Layout only extraction (prompt_layout_only_en)"
        },
        "usage": {
            "file_upload": "Send as multipart/form-data with 'file' or 'files' field",
            "python_base64": "Send as form-data with 'image' or 'images' field (base64 encoded)"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        return {
            "status": "healthy",
            "vllm_ip": VLLM_IP,
            "vllm_port": VLLM_PORT,
            "model_name": MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/get-text")
async def get_text(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Extract text from image using OCR
    Uses prompt mode: prompt_ocr
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with extracted text
    """
    start_time = time.time()
    prompt_mode = "prompt_ocr"
    
    try:
        pil_images = []
        is_batch = False
        
        # Smart detection of input type
        if file:
            # Single file upload
            content = await file.read()
            pil_images = [decode_image(content)]
            logger.info(f"Processing single file upload: {file.filename}")
            
        elif files:
            # Batch file upload
            is_batch = True
            for f in files:
                content = await f.read()
                pil_images.append(decode_image(content))
            logger.info(f"Processing batch file upload: {len(files)} files")
            
        elif image:
            # Single base64 (from Python client)
            pil_images = [decode_image(image)]
            logger.info("Processing single base64 image")
            
        elif images:
            # Batch base64 (from Python client)
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Get prompt from mode
        prompt = dict_promptmode_to_prompt.get(prompt_mode)
        if not prompt:
            raise HTTPException(status_code=500, detail=f"Prompt mode '{prompt_mode}' not found")
        
        # Process images
        results = []
        for img in pil_images:
            response = inference_with_vllm(
                img,
                prompt,
                ip=VLLM_IP,
                port=VLLM_PORT,
                temperature=0.1,
                top_p=0.9,
                model_name=MODEL_NAME,
            )
            results.append(response)
        
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch text extraction completed in {processing_time:.2f}s")
            return {
                "mode": "text_extraction_batch",
                "prompt_mode": prompt_mode,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            logger.info(f"Text extraction completed in {processing_time:.2f}s")
            return {
                "filename": getattr(file, 'filename', None),
                "mode": "text_extraction",
                "prompt_mode": prompt_mode,
                "result": results[0],
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/extract")
async def extract(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Extract full layout and content from image
    Uses prompt mode: prompt_layout_all_en
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with layout and content extraction
    """
    start_time = time.time()
    prompt_mode = "prompt_layout_all_en"
    
    try:
        pil_images = []
        is_batch = False
        
        # Smart detection of input type
        if file:
            # Single file upload
            content = await file.read()
            pil_images = [decode_image(content)]
            logger.info(f"Processing single file upload: {file.filename}")
            
        elif files:
            # Batch file upload
            is_batch = True
            for f in files:
                content = await f.read()
                pil_images.append(decode_image(content))
            logger.info(f"Processing batch file upload: {len(files)} files")
            
        elif image:
            # Single base64 (from Python client)
            pil_images = [decode_image(image)]
            logger.info("Processing single base64 image")
            
        elif images:
            # Batch base64 (from Python client)
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Get prompt from mode
        prompt = dict_promptmode_to_prompt.get(prompt_mode)
        if not prompt:
            raise HTTPException(status_code=500, detail=f"Prompt mode '{prompt_mode}' not found")
        
        # Process images
        results = []
        for img in pil_images:
            response = inference_with_vllm(
                img,
                prompt,
                ip=VLLM_IP,
                port=VLLM_PORT,
                temperature=0.1,
                top_p=0.9,
                model_name=MODEL_NAME,
            )
            results.append(response)
        
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch full extraction completed in {processing_time:.2f}s")
            return {
                "mode": "full_extraction_batch",
                "prompt_mode": prompt_mode,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            logger.info(f"Full extraction completed in {processing_time:.2f}s")
            return {
                "filename": getattr(file, 'filename', None),
                "mode": "full_extraction",
                "prompt_mode": prompt_mode,
                "result": results[0],
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/get-layout")
async def get_layout(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Extract layout information only from image
    Uses prompt mode: prompt_layout_only_en
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with layout information
    """
    start_time = time.time()
    prompt_mode = "prompt_layout_only_en"
    
    try:
        pil_images = []
        is_batch = False
        
        # Smart detection of input type
        if file:
            # Single file upload
            content = await file.read()
            pil_images = [decode_image(content)]
            logger.info(f"Processing single file upload: {file.filename}")
            
        elif files:
            # Batch file upload
            is_batch = True
            for f in files:
                content = await f.read()
                pil_images.append(decode_image(content))
            logger.info(f"Processing batch file upload: {len(files)} files")
            
        elif image:
            # Single base64 (from Python client)
            pil_images = [decode_image(image)]
            logger.info("Processing single base64 image")
            
        elif images:
            # Batch base64 (from Python client)
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Get prompt from mode
        prompt = dict_promptmode_to_prompt.get(prompt_mode)
        if not prompt:
            raise HTTPException(status_code=500, detail=f"Prompt mode '{prompt_mode}' not found")
        
        # Process images
        results = []
        for img in pil_images:
            response = inference_with_vllm(
                img,
                prompt,
                ip=VLLM_IP,
                port=VLLM_PORT,
                temperature=0.1,
                top_p=0.9,
                model_name=MODEL_NAME,
            )
            results.append(response)
        
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch layout extraction completed in {processing_time:.2f}s")
            return {
                "mode": "layout_extraction_batch",
                "prompt_mode": prompt_mode,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            logger.info(f"Layout extraction completed in {processing_time:.2f}s")
            return {
                "filename": getattr(file, 'filename', None),
                "mode": "layout_extraction",
                "prompt_mode": prompt_mode,
                "result": results[0],
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9667, log_level="info")
