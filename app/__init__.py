from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Medical Imaging API",
        description="API for processing medical images using AI models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Ensure required directories exist
    settings.ensure_directories()
    
    return app