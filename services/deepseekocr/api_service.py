from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import time
import os
import logging
import base64
import json
from typing import Optional, List
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="DeepSeek-OCR API Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# Initialize vLLM
vllm_llm = None


# Helper functions
def decode_image(data):
    """
    Smart decoder - handles both base64 strings and raw bytes
    Returns PIL Image in RGB mode
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
        
        # Open with PIL and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def get_vllm_engine():
    """Initialize vLLM engine"""
    global vllm_llm
    if vllm_llm is None:
        try:
            logger.info(f"Initializing vLLM engine with model: {MODEL_NAME}")
            logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
            logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
            
            # Initialize vLLM engine with NGramPerReqLogitsProcessor
            vllm_llm = LLM(
                model=MODEL_NAME,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                trust_remote_code=True,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}", exc_info=True)
            raise
    return vllm_llm


def create_sampling_params(temperature=0.0, max_tokens=None):
    """Create sampling parameters for DeepSeek-OCR"""
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    logger.info("Starting DeepSeek-OCR API service...")
    try:
        get_vllm_engine()
        logger.info("DeepSeek-OCR API service ready")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        # Don't fail completely, allow service to start for health checks
        pass


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "DeepSeek-OCR API",
        "model": MODEL_NAME,
        "backend": "vllm-engine",
        "initialized": vllm_llm is not None,
        "endpoints": {
            "/get-text": "Free OCR text extraction (file upload or base64)",
            "/extract": "Convert document to markdown with grounding (file upload or base64)",
            "/get-table-structure": "Convert table to html"
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
        is_healthy = vllm_llm is not None
        
        return {
            "status": "healthy" if is_healthy else "initializing",
            "model_name": MODEL_NAME,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_tokens": MAX_TOKENS,
            "vllm_initialized": vllm_llm is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/get-text")
async def ocr(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Universal Free OCR endpoint
    Prompt: "<image>\nFree OCR."
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with extracted text
    """
    start_time = time.time()
    
    try:
        # Get vLLM engine
        llm = get_vllm_engine()
        
        if llm is None:
            raise HTTPException(status_code=503, detail="Service not initialized yet")
        
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
        
        # Prepare prompt
        prompt = "<image>\nFree OCR."
        
        # Prepare model inputs
        model_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            }
            for img in pil_images
        ]
        
        # Create sampling parameters
        sampling_params = create_sampling_params()
        
        # Generate outputs
        model_outputs = llm.generate(model_inputs, sampling_params)
        
        # Extract results
        processing_time = time.time() - start_time
        
        if is_batch:
            results = [output.outputs[0].text for output in model_outputs]
            logger.info(f"Batch OCR completed in {processing_time:.2f}s")
            return {
                "mode": "free_ocr_batch",
                "prompt": prompt,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            result = model_outputs[0].outputs[0].text
            logger.info(f"OCR completed in {processing_time:.2f}s")
            return {
                "mode": "free_ocr",
                "prompt": prompt,
                "result": result,
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing OCR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/extract")
async def markdown(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Universal Markdown conversion endpoint with grounding
    Prompt: "<image>\n<|grounding|>Convert the document to markdown."
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with markdown conversion and grounding
    """
    start_time = time.time()
    
    try:
        # Get vLLM engine
        llm = get_vllm_engine()
        
        if llm is None:
            raise HTTPException(status_code=503, detail="Service not initialized yet")
        
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
        
        # Prepare prompt
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        # Prepare model inputs
        model_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            }
            for img in pil_images
        ]
        
        # Create sampling parameters
        sampling_params = create_sampling_params()
        
        # Generate outputs
        model_outputs = llm.generate(model_inputs, sampling_params)
        
        # Extract results
        processing_time = time.time() - start_time
        
        if is_batch:
            results = [output.outputs[0].text for output in model_outputs]
            logger.info(f"Batch markdown extraction completed in {processing_time:.2f}s")
            return {
                "mode": "markdown_grounding_batch",
                "prompt": prompt,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            result = model_outputs[0].outputs[0].text
            logger.info(f"Markdown extraction completed in {processing_time:.2f}s")
            return {
                "mode": "markdown_grounding",
                "prompt": prompt,
                "result": result,
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing markdown extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/get-table-structure")
async def get_table_structure(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Universal Markdown conversion endpoint with grounding
    Prompt: "<image>\n<|grounding|>Convert the document to markdown."
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with markdown conversion and grounding
    """
    start_time = time.time()
    
    try:
        # Get vLLM engine
        llm = get_vllm_engine()
        
        if llm is None:
            raise HTTPException(status_code=503, detail="Service not initialized yet")
        
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
        
        # Prepare prompt
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        # Prepare model inputs
        model_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            }
            for img in pil_images
        ]
        
        # Create sampling parameters
        sampling_params = create_sampling_params()
        
        # Generate outputs
        model_outputs = llm.generate(model_inputs, sampling_params)
        
        # Extract results
        processing_time = time.time() - start_time
        
        if is_batch:
            results = [output.outputs[0].text for output in model_outputs]
            logger.info(f"Batch markdown extraction completed in {processing_time:.2f}s")
            return {
                "mode": "markdown_grounding_batch",
                "prompt": prompt,
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            result = model_outputs[0].outputs[0].text
            logger.info(f"Markdown extraction completed in {processing_time:.2f}s")
            return {
                "mode": "markdown_grounding",
                "prompt": prompt,
                "result": result,
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing markdown extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9666, log_level="info")