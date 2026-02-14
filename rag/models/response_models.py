"""API response models."""

from typing import Optional, Any, Dict
from pydantic import BaseModel
import time


class APIResponse(BaseModel):
    """Standard API response model."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __init__(self, **data):
        """Initialize with automatic timestamp if not provided."""
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        super().__init__(**data)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "example"},
                "error": None,
                "timestamp": "2024-01-01 12:00:00"
            }
        }


def success_response(data: Any) -> Dict:
    """Create a success response."""
    return APIResponse(success=True, data=data).model_dump()


def error_response(error: str, status_code: int = 400) -> tuple:
    """Create an error response with status code."""
    return APIResponse(success=False, error=error).model_dump(), status_code
