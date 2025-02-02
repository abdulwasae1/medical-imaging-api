import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional
from ..config import settings

def decode_image(image_data: str) -> np.ndarray:
    """
    Decode base64 image to numpy array
    
    Args:
        image_data (str): Base64 encoded image string
        
    Returns:
        np.ndarray: Decoded image array in BGR format
        
    Raises:
        ValueError: If image decoding fails
    """
    try:
        img_bytes = base64.b64decode(image_data)
        img_buf = BytesIO(img_bytes)
        img_pil = Image.open(img_buf)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")

def encode_image(image: np.ndarray) -> str:
    """
    Encode numpy array to base64 string
    
    Args:
        image (np.ndarray): Image array in BGR format
        
    Returns:
        str: Base64 encoded image string
        
    Raises:
        ValueError: If image encoding fails
    """
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")

def preprocess_brain_tumor_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for brain tumor detection
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        np.ndarray: Preprocessed image ready for model inference
    """
    img = cv2.resize(img, (240, 240))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def validate_image(img: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image dimensions and content
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        Tuple[bool, str]: Validation result and error message if any
    """
    if img is None:
        return False, "Invalid image data"
    if len(img.shape) != 3:
        return False, "Image must be in color (3 channels)"
    if img.shape[2] != 3:
        return False, "Image must be in RGB format"
    return True, ""

def save_debug_image(img: np.ndarray, filename: str) -> Optional[str]:
    """
    Save image for debugging purposes
    
    Args:
        img (np.ndarray): Image to save
        filename (str): Desired filename
        
    Returns:
        Optional[str]: Path to saved image or None if failed
    """
    if settings.DEBUG:
        try:
            path = f"{settings.UPLOAD_DIR}/{filename}"
            cv2.imwrite(path, img)
            return path
        except Exception:
            return None
    return None