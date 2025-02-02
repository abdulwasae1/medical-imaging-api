import cv2
import numpy as np
import tensorflow as tf
from ..config import settings
from ..utils.image_processing import preprocess_brain_tumor_image, Timer

class DisplayTumor:
    def __init__(self):
        self.img = None
        self.curr_img = None
        self.thresh = None
        self.ret = None

    def readImage(self, img):
        """Initialize image processing"""
        self.img = np.array(img)
        self.curr_img = np.array(img)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def removeNoise(self):
        """Remove noise from image using morphological operations"""
        kernel = np.ones((3, 3), np.uint8)
        self.curr_img = cv2.morphologyEx(
            self.thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    def displayTumor(self):
        """Process image to highlight tumor regions"""
        kernel = np.ones((3, 3), np.uint8)
        
        # Background segmentation
        sure_bg = cv2.dilate(self.curr_img, kernel, iterations=3)
        
        # Foreground extraction
        dist_transform = cv2.distanceTransform(self.curr_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(
            dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(self.img, markers)
        self.img[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        self.curr_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def getImage(self):
        """Return processed image"""
        return self.curr_img

class BrainTumorService:
    def __init__(self):
        """Initialize the service with the brain tumor model"""
        try:
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load brain tumor model: {str(e)}")

    def detect_tumor(self, img: np.ndarray) -> dict:
        """
        Detect brain tumor in the image
        
        Args:
            img: Input image as numpy array
            
        Returns:
            dict containing detection results and processed image
        """
        with Timer() as t:
            try:
                # Preprocess image
                preprocessed_img = preprocess_brain_tumor_image(img)
                
                # Run inference
                prediction = self.model.predict(preprocessed_img, verbose=0)
                
                # Process image to highlight tumor regions
                display_tumor = DisplayTumor()
                display_tumor.readImage(img)
                display_tumor.removeNoise()
                display_tumor.displayTumor()
                processed_img = display_tumor.getImage()
                
                # Prepare results
                is_tumor = bool(np.argmax(prediction) == 1)
                confidence = float(prediction[0][np.argmax(prediction)])
                
                return {
                    "prediction_results": {
                        "tumor_detected": is_tumor,
                        "confidence": confidence,
                        "probability_scores": {
                            "normal": float(prediction[0][0]),
                            "tumor": float(prediction[0][1])
                        }
                    },
                    "processed_image": processed_img,
                    "processing_time": t.elapsed,
                    "status": "success"
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "processing_time": t.elapsed
                }