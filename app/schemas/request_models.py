from pydantic import BaseModel, validator, Field
from typing import List, Optional
import base64

class BoundingBox(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    label: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "x": 0.5,
                "y": 0.5,
                "width": 0.2,
                "height": 0.2,
                "confidence": 0.95,
                "label": "fracture"
            }
        }

class ImageRequest(BaseModel):
    image_data: str
    bounding_boxes: Optional[List[BoundingBox]] = None
    
    @validator('image_data')
    def validate_image_data(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_string",
                "bounding_boxes": [
                    {
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                        "confidence": 0.95,
                        "label": "fracture"
                    }
                ]
            }
        }

class DetectionResponse(BaseModel):
    processed_image: str
    tumor_detected: Optional[bool] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "processed_image": "base64_encoded_processed_image",
                "tumor_detected": True,
                "confidence": 0.95,
                "error": None
            }
        }

class ErrorResponse(BaseModel):
    detail: str