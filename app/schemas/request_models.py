from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    label: str

class ImageRequest(BaseModel):
    image_data: str
    bounding_boxes: Optional[List[BoundingBox]] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_string",
                "bounding_boxes": [
                    {
                        "x": 0.1,
                        "y": 0.1,
                        "width": 0.2,
                        "height": 0.2,
                        "confidence": 0.95,
                        "label": "fracture"
                    }
                ]
            }
        }

class ProcessingResponse(BaseModel):
    processed_image: str
    prediction_results: dict
    processing_time: float
    status: str = "success"