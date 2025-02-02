import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in development
if os.path.exists('.env'):
    load_dotenv()

class Settings:
    # Base Settings
    BASE_DIR = Path(__file__).resolve().parent.parent
    API_V1_STR = "/api/v1"
    PROJECT_NAME = "Medical Imaging API"
    
    # Model Settings
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models/brain_tumor_detector.h5"))
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    
    # Server Settings
    PORT = int(os.getenv("PORT", 8000))
    RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
    
    # Image Settings
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Security Settings
    CORS_ORIGINS = [
        "http://localhost",
        "http://localhost:8000",
        "https://localhost",
        "https://localhost:8000",
    ]

settings = Settings()
