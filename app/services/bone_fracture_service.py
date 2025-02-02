import cv2
import numpy as np
from ..utils.exceptions import ImageProcessingError
from ..schemas.request_models import BoundingBox
from typing import List

class BoneFractureService:
    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: List[BoundingBox]) -> np.ndarray:
        try:
            img_height, img_width = image.shape[:2]
            
            for box in boxes:
                x1 = int(box.x * img_width)
                y1 = int(box.y * img_height)
                x2 = int((box.x + box.width) * img_width)
                y2 = int((box.y + box.height) * img_height)
                
                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{box.label} {box.confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image
        except Exception as e:
            raise ImageProcessingError(f"Error drawing bounding boxes: {str(e)}")
