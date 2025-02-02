from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app import create_app
from app.services.brain_tumor_service import BrainTumorService
from app.schemas.request_models import ImageRequest, DetectionResponse, ErrorResponse
from app.utils.image_processing import decode_image, encode_image, validate_image, save_debug_image
from app.config import settings
import uvicorn
import os
import logging
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = create_app()
brain_tumor_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global brain_tumor_service
    try:
        brain_tumor_service = BrainTumorService()
        logger.info("Successfully initialized brain tumor service")
    except Exception as e:
        logger.error(f"Failed to initialize brain tumor service: {str(e)}")
        raise

@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Medical Imaging API is running",
        "version": "1.0.0"
    }

@app.post(
    "/api/detect-brain-tumor", 
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def detect_brain_tumor(image_request: ImageRequest):
    """
    Detect brain tumors in the provided image
    
    Args:
        image_request (ImageRequest): Request containing base64 encoded image
        
    Returns:
        DetectionResponse: Processed image and detection results
        
    Raises:
        HTTPException: For invalid requests or processing errors
    """
    try:
        # Decode and validate image
        image = decode_image(image_request.image_data)
        valid, error_message = validate_image(image)
        if not valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Process image
        result = brain_tumor_service.detect_tumor(image)
        
        if settings.DEBUG:
            save_debug_image(result["processed_image"], "debug_tumor.jpg")
        
        # Prepare response
        return DetectionResponse(
            processed_image=encode_image(result["processed_image"]),
            tumor_detected=result["tumor_detected"],
            confidence=result["confidence"]
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/process-bone-fracture",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def process_bone_fracture(image_request: ImageRequest):
    """
    Process bone fracture images with provided bounding boxes
    
    Args:
        image_request (ImageRequest): Request containing image and bounding boxes
        
    Returns:
        DetectionResponse: Processed image with drawn bounding boxes
        
    Raises:
        HTTPException: For invalid requests or processing errors
    """
    try:
        # Validate request
        if not image_request.bounding_boxes:
            raise HTTPException(
                status_code=400,
                detail="Bounding boxes are required for bone fracture processing"
            )
        
        # Decode and validate image
        image = decode_image(image_request.image_data)
        valid, error_message = validate_image(image)
        if not valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Draw bounding boxes
        for box in image_request.bounding_boxes:
            x1 = int(box.x * image.shape[1])
            y1 = int(box.y * image.shape[0])
            x2 = int((box.x + box.width) * image.shape[1])
            y2 = int((box.y + box.height) * image.shape[0])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, 
                f"{box.label} {box.confidence:.2f}", 
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        if settings.DEBUG:
            save_debug_image(image, "debug_fracture.jpg")
        
        return DetectionResponse(
            processed_image=encode_image(image)
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)