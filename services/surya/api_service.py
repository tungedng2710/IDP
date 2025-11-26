from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import time
import sys
import logging
import numpy as np
import base64
from typing import List, Optional, Union

# Add lib to path
sys.path.append('/app/lib/')

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Surya OCR API Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictors
foundation_predictor: Optional[FoundationPredictor] = None
recognition_predictor: Optional[RecognitionPredictor] = None
detection_predictor: Optional[DetectionPredictor] = None


# Helper functions
def decode_image(data: Union[str, bytes]) -> Image.Image:
    """
    Smart decoder - handles both base64 strings and raw bytes
    Automatically detects the format
    """
    try:
        # Case 1: It's a base64 string
        if isinstance(data, str):
            # Remove data URI prefix if present (e.g., "data:image/png;base64,")
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


def process_images(data: Union[str, List[str], bytes, List[bytes]]) -> List[Image.Image]:
    """
    Smart processor - handles single or batch, base64 or bytes
    Returns list of PIL Images
    """
    if isinstance(data, list):
        # Batch processing
        return [decode_image(item) for item in data]
    else:
        # Single image
        return [decode_image(data)]


def format_bbox_text_results(recognition_predictions):
    """
    Format recognition predictions to bbox-text format
    
    Returns:
        List of dicts with {'bbox': [x1, y1, x2, y2], 'text': str}
    """
    formatted_results = []
    
    for pred in recognition_predictions:
        image_results = []
        
        # Extract text lines from prediction
        for text_line in pred.text_lines:
            bbox = text_line.bbox
            text = text_line.text
            
            # Convert bbox to [left, top, right, bottom] format
            # bbox is typically [x1, y1, x2, y2] or similar
            bbox_formatted = [
                int(bbox[0]),  # left (x1)
                int(bbox[1]),  # top (y1)
                int(bbox[2]),  # right (x2)
                int(bbox[3])   # bottom (y2)
            ]
            
            image_results.append({
                'bbox': bbox_formatted,
                'text': text
            })
        
        formatted_results.append(image_results)
    
    return formatted_results


@app.on_event("startup")
async def startup_event():
    """Initialize predictors on startup"""
    global foundation_predictor, recognition_predictor, detection_predictor
    
    logger.info("Initializing predictors...")
    try:
        foundation_predictor = FoundationPredictor()
        logger.info("Foundation predictor initialized")
        
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        logger.info("Recognition predictor initialized")
        
        detection_predictor = DetectionPredictor()
        logger.info("Detection predictor initialized")
        
        logger.info("All predictors initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictors: {str(e)}", exc_info=True)
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Surya OCR API",
        "endpoints": {
            "/get-lines": "Text detection (file upload or base64)",
            "/get-text": "Text recognition (file upload or base64)",
            "/get-bbox-text": "Combined detection + recognition (file upload or base64)"
        },
        "usage": {
            "file_upload": "Send as multipart/form-data with 'file' or 'files' field",
            "python_base64": "Send as JSON with 'image' or 'images' field (base64 encoded)"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        predictor_status = {
            "foundation_predictor": foundation_predictor is not None,
            "recognition_predictor": recognition_predictor is not None,
            "detection_predictor": detection_predictor is not None
        }
        
        all_ready = all(predictor_status.values())
        
        return {
            "status": "healthy" if all_ready else "initializing",
            "predictors": predictor_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/get-lines")
async def detect(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Universal text detection endpoint
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 JSON: 'image' (single) or 'images' (batch as JSON string)
    
    Automatically detects input type and processes accordingly
    """
    start_time = time.time()
    
    if detection_predictor is None:
        raise HTTPException(status_code=503, detail="Detection predictor not initialized")
    
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
            # Expecting JSON array string
            import json
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Run detection
        detection_predictions = detection_predictor(pil_images)
        
        # Format response
        processing_time = time.time() - start_time
        
        if is_batch:
            results = [pred.model_dump() for pred in detection_predictions]
            logger.info(f"Batch detection completed in {processing_time:.2f}s")
            return {
                "mode": "text_detection_batch",
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            result = detection_predictions[0].model_dump()
            logger.info(f"Detection completed in {processing_time:.2f}s")
            return {
                "mode": "text_detection",
                "result": result,
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/get-text")
async def recognize(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Universal text recognition endpoint
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 JSON: 'image' (single) or 'images' (batch as JSON string)
    
    Automatically detects input type and processes accordingly
    """
    start_time = time.time()
    
    if recognition_predictor is None or detection_predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Recognition or detection predictor not initialized"
        )
    
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
            import json
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Run recognition
        recognition_predictions = recognition_predictor(
            pil_images, 
            det_predictor=detection_predictor
        )
        
        # Format response
        processing_time = time.time() - start_time
        
        if is_batch:
            results = [pred.model_dump() for pred in recognition_predictions]
            logger.info(f"Batch recognition completed in {processing_time:.2f}s")
            return {
                "mode": "text_recognition_batch",
                "count": len(results),
                "results": results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(results)
            }
        else:
            result = recognition_predictions[0].model_dump()
            logger.info(f"Recognition completed in {processing_time:.2f}s")
            return {
                "mode": "text_recognition",
                "result": result,
                "processing_time": processing_time
            }
        
    except ValueError as e:
        logger.error(f"Invalid image data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/get-bbox-text")
async def get_bbox_text(
    # For file upload (single or multiple)
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    # For JSON/Python client (single or batch)
    image: Optional[str] = Form(None),
    images: Optional[str] = Form(None)
):
    """
    Combined detection + recognition endpoint
    Returns bbox-text pairs in simple format
    
    Accepts either:
    1. File upload: 'file' (single) or 'files' (batch)
    2. Base64 JSON: 'image' (single) or 'images' (batch as JSON string)
    
    Returns:
        Single image: List[Dict] with {'bbox': [x1, y1, x2, y2], 'text': str}
        Batch: List of above lists
    """
    start_time = time.time()
    
    if recognition_predictor is None or detection_predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Recognition or detection predictor not initialized"
        )
    
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
            import json
            is_batch = True
            images_list = json.loads(images)
            pil_images = [decode_image(img) for img in images_list]
            logger.info(f"Processing batch base64: {len(images_list)} images")
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="No image data provided. Send 'file'/'files' (upload) or 'image'/'images' (base64)"
            )
        
        # Run recognition (includes detection)
        recognition_predictions = recognition_predictor(
            pil_images, 
            det_predictor=detection_predictor
        )
        
        # Format to bbox-text pairs
        formatted_results = format_bbox_text_results(recognition_predictions)
        
        # Format response
        processing_time = time.time() - start_time
        
        if is_batch:
            logger.info(f"Batch bbox-text extraction completed in {processing_time:.2f}s")
            return {
                "mode": "bbox_text_batch",
                "count": len(formatted_results),
                "results": formatted_results,
                "processing_time": processing_time,
                "avg_time_per_image": processing_time / len(formatted_results)
            }
        else:
            logger.info(f"Bbox-text extraction completed in {processing_time:.2f}s")
            return {
                "mode": "bbox_text",
                "result": formatted_results[0],
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
    uvicorn.run(app, host="0.0.0.0", port=9669, log_level="info")