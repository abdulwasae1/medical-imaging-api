import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Base settings
    PROJECT_NAME = "Medical Imaging API"
    VERSION = "1.0.0"
    API_V1_STR = "/api"
    
    # Environment settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/brain_tumor_detector.h5")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    
    # Render specific
    IS_PRODUCTION = os.getenv("RENDER", "False").lower() == "true"
    PORT = int(os.getenv("PORT", "8000"))
    HOST = "0.0.0.0" if IS_PRODUCTION else "127.0.0.1"
    
    # Image settings
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Model settings
    TUMOR_CONFIDENCE_THRESHOLD = 0.5
    FRACTURE_CONFIDENCE_THRESHOLD = 0.3

settings = Settings()