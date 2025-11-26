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
from vllm import LLM
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
# Create FastAPI app
app = FastAPI(title="MinerU API Service", version="1.0.0")
llm = LLM(
    model="opendatalab/MinerU2.5-2509-1.2B",
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    logits_processors=[MinerULogitsProcessor]  # if vllm>=0.10.1
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "opendatalab/MinerU2.5-2509-1.2B")

# Initialize MinerU client (will connect to vLLM server)
mineru_client = None


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


def get_mineru_client():
    """Initialize MinerU client with vLLM backend"""
    global mineru_client
    if mineru_client is None:
        try:
            logger.info(f"Initializing MinerU client with vLLM backend at {VLLM_BASE_URL}")
            # Use remote vLLM server
            mineru_client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=llm
            )
            logger.info("MinerU client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MinerU client: {str(e)}")
            raise
    return mineru_client


@app.on_event("startup")
async def startup_event():
    """Initialize client on startup"""
    get_mineru_client()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "MinerU API",
        "vllm_endpoint": VLLM_BASE_URL,
        "model": MODEL_NAME,
        "endpoints": {
            "/get-layout": "Layout detection only",
            "/extract": "Two-step full extraction (layout + content)"
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
            "vllm_base_url": VLLM_BASE_URL,
            "model_name": MODEL_NAME,
            "client_initialized": mineru_client is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


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
    Uses client.layout_detect(image)
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with layout detection results
    """
    start_time = time.time()
    
    try:
        # Get MinerU client
        client = get_mineru_client()
        
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
        
        # Process images - layout detection
        results = []
        for img in pil_images:
            layout_result = client.layout_detect(img)
            results.append(layout_result)
        
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch layout detection completed in {processing_time:.2f}s")
            return {
                "mode": "layout_detection_batch",
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            logger.info(f"Layout detection completed in {processing_time:.2f}s")
            return {
                "filename": getattr(file, 'filename', None),
                "mode": "layout_detection",
                "result": results[0],
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing layout detection: {str(e)}", exc_info=True)
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
    Extract full layout and content from image using two-step extraction
    Uses client.two_step_extract(image)
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 form-data: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        JSON with extracted blocks (layout + content)
    """
    start_time = time.time()
    
    try:
        # Get MinerU client
        client = get_mineru_client()
        
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
        
        # Process images - two-step extraction
        results = []
        for img in pil_images:
            extracted_blocks = client.two_step_extract(img)
            results.append(extracted_blocks)
        
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch two-step extraction completed in {processing_time:.2f}s")
            return {
                "mode": "two_step_extraction_batch",
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            logger.info(f"Two-step extraction completed in {processing_time:.2f}s")
            return {
                "filename": getattr(file, 'filename', None),
                "mode": "two_step_extraction",
                "result": results[0],
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9668, log_level="info")