from fastapi import HTTPException
from typing import Any

class ImageProcessingError(HTTPException):
    def __init__(self, detail: Any = None):
        super().__init__(status_code=422, detail=detail or "Error processing image")

class ModelInferenceError(HTTPException):
    def __init__(self, detail: Any = None):
        super().__init__(status_code=500, detail=detail or "Error during model inference")