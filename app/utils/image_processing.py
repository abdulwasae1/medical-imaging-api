import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from .exceptions import ImageProcessingError
from ..config import settings

def validate_image_size(image_data: str) -> bool:
    """Validate image size before processing"""
    size = len(image_data) * 3/4  # Base64 to binary ratio
    return size <= settings.MAX_IMAGE_SIZE

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        if not validate_image_size(image_data):
            raise ImageProcessingError("Image size exceeds maximum limit")
            
        img_bytes = base64.b64decode(image_data)
        img_buf = BytesIO(img_bytes)
        img_pil = Image.open(img_buf)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ImageProcessingError(f"Error decoding image: {str(e)}")

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        raise ImageProcessingError(f"Error encoding image: {str(e)}")

def preprocess_brain_tumor_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for brain tumor detection"""
    try:
        img = cv2.resize(img, (240, 240))
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ImageProcessingError(f"Error preprocessing image: {str(e)}")