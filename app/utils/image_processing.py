import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional
import time
from ..config import settings

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        img_bytes = base64.b64decode(image_data)
        img_buf = BytesIO(img_bytes)
        img_pil = Image.open(img_buf)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    try:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")

def preprocess_brain_tumor_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for brain tumor detection"""
    try:
        img = cv2.resize(img, (240, 240))
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def draw_boxes(
    image: np.ndarray,
    boxes: list,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw bounding boxes on image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    for box in boxes:
        x1 = int(box.x * w)
        y1 = int(box.y * h)
        x2 = int((box.x + box.width) * w)
        y2 = int((box.y + box.height) * h)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{box.label} {box.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        
        cv2.rectangle(img, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
    
    return img

class Timer:
    """Context manager for timing code execution"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

def validate_image_size(image_data: str) -> bool:
    """Validate image size against MAX_IMAGE_SIZE"""
    size = len(image_data) * 3/4  # Base64 encoding increases size by 4/3
    return size <= settings.MAX_IMAGE_SIZE