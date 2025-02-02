import cv2
import numpy as np
import tensorflow as tf
from ..utils.image_processing import preprocess_brain_tumor_image
from ..utils.exceptions import ModelInferenceError
from ..config import settings

class BrainTumorService:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)
        except Exception as e:
            raise ModelInferenceError(f"Error loading brain tumor model: {str(e)}")

    def detect_tumor(self, img: np.ndarray):
        try:
            preprocessed_img = preprocess_brain_tumor_image(img)
            prediction = self.model.predict(preprocessed_img)
            
            display_tumor = DisplayTumor()
            display_tumor.readImage(img)
            display_tumor.removeNoise()
            display_tumor.displayTumor()
            processed_img = display_tumor.getImage()
            
            return {
                "tumor_detected": bool(np.argmax(prediction) == 1),
                "confidence": float(prediction[0][np.argmax(prediction)]),
                "processed_image": processed_img
            }
        except Exception as e:
            raise ModelInferenceError(f"Error during tumor detection: {str(e)}")

class DisplayTumor:
    def readImage(self, img):
        self.img = np.array(img)
        self.curr_img = np.array(img)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def removeNoise(self):
        kernel = np.ones((3, 3), np.uint8)
        self.curr_img = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    def displayTumor(self):
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(self.curr_img, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(self.curr_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(self.img, markers)
        self.img[markers == -1] = [255, 0, 0]
        
        self.curr_img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)

    def getImage(self):
        return self.curr_img