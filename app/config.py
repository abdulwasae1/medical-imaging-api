import os
from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache

class Settings(BaseSettings):
    # Model settings
    MODEL_PATH: str = "./models/brain_tumor_detector.h5"
    
    # Upload settings
    UPLOAD_DIR: str = "./uploads"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"png", "jpg", "jpeg"}
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path("./models").mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()