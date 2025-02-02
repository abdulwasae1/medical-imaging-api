import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.brain_tumor_service import BrainTumorService
from app.schemas.request_models import ImageRequest
from app.utils.image_processing import (
    decode_image, encode_image, validate_image_size, draw_boxes
)
from app.config import settings
import uvicorn
import time
from app.schemas.request_models import ImageRequest, ImageResponse


# Create required directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for medical image processing and analysis",
    version=settings.VERSION,
    docs_url="/docs" if not settings.IS_PRODUCTION else None,
    redoc_url="/redoc" if not settings.IS_PRODUCTION else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
brain_tumor_service = BrainTumorService()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    return {
        "message": "Medical Imaging API is running",
        "environment": settings.RAILWAY_ENVIRONMENT,
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.post("/api/detect-brain-tumor", response_model=ImageResponse)
async def detect_brain_tumor(image_request: ImageRequest):
    try:
        # Decode the base64 image
        image = decode_image(image_request.image_data)
        
        # Process the image using the brain tumor service
        result = brain_tumor_service.detect_tumor(image)
        
        # Prepare response
        return ImageResponse(
            processed_image=encode_image(result["processed_image"]),
            detection_result={
                "tumor_detected": result["tumor_detected"],
                "confidence": result["confidence"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-bone-fracture", response_model=ImageResponse)
async def process_bone_fracture(image_request: ImageRequest):
    """
    Process image with bone fracture bounding boxes
    
    Args:
        image_request: ImageRequest object containing base64 encoded image and bounding boxes
        
    Returns:
        ProcessingResponse object with processed image
    """
    try:
        # Validate image size
        if not validate_image_size(image_request.image_data):
            raise HTTPException(
                status_code=400,
                detail=f"Image size exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE/1024/1024}MB"
            )
        
        # Validate bounding boxes
        if not image_request.bounding_boxes:
            raise HTTPException(
                status_code=400,
                detail="Bounding boxes are required for bone fracture processing"
            )
        
        start_time = time.time()
        
        # Decode the base64 image
        image = decode_image(image_request.image_data)
        
        # Draw bounding boxes on the image
        processed_image = draw_boxes(
            image,
            image_request.bounding_boxes,
            color=(0, 255, 0),
            thickness=2
        )
        
        # Prepare response
        result = {
            "processed_image": encode_image(processed_image),
            "prediction_results": {
                "num_detections": len(image_request.bounding_boxes),
                "detections": [
                    {
                        "label": box.label,
                        "confidence": box.confidence,
                        "bbox": {
                            "x": box.x,
                            "y": box.y,
                            "width": box.width,
                            "height": box.height
                        }
                    }
                    for box in image_request.bounding_boxes
                ]
            },
            "processing_time": time.time() - start_time,
            "status": "success"
        }
        
        return ImageResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=not settings.IS_PRODUCTION,
        workers=4 if settings.IS_PRODUCTION else 1
    )