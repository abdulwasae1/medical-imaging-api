from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    label: str

    class Config:
        schema_extra = {
            "example": {
                "x": 0.1,
                "y": 0.2,
                "width": 0.3,
                "height": 0.4,
                "confidence": 0.95,
                "label": "fracture"
            }
        }

class ImageRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    bounding_boxes: Optional[List[BoundingBox]] = Field(None, description="Bounding boxes for bone fracture detection")

    class Config:
        schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_string",
                "bounding_boxes": None
            }
        }

class ImageResponse(BaseModel):
    processed_image: str = Field(..., description="Base64 encoded processed image")
    detection_result: Optional[dict] = Field(None, description="Detection results")