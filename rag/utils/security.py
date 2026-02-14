"""Security utilities for file handling and validation."""

import os
import logging
from pathlib import Path
from typing import Optional
from werkzeug.utils import secure_filename
from rag.config.constants import ALLOWED_EXTENSIONS, MAX_FILENAME_LENGTH
from rag.utils.exceptions import SecurityError, ValidationError


def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed
    """
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and
        len(filename) <= MAX_FILENAME_LENGTH
    )


def secure_upload_filename(filename: str) -> str:
    """
    Securely handle file upload filename.
    
    Args:
        filename: Original filename
        
    Returns:
        Secured filename
        
    Raises:
        ValidationError: If filename is invalid
    """
    if not filename or not allowed_file(filename):
        raise ValidationError(
            f"Invalid file type or filename. Allowed extensions: {ALLOWED_EXTENSIONS}"
        )
    
    # Use werkzeug's secure_filename
    secured = secure_filename(filename)
    
    # Additional sanitization
    secured = secured.replace('..', '')
    
    return secured


def validate_path(base_dir: Path, user_path: str) -> Path:
    """
    Ensure user_path is within base_dir to prevent path traversal attacks.
    
    Args:
        base_dir: Base directory that paths must be within
        user_path: User-provided path to validate
        
    Returns:
        Validated absolute path
        
    Raises:
        SecurityError: If path traversal is detected
    """
    base_dir = Path(base_dir).resolve()
    full_path = (base_dir / user_path).resolve()
    
    # Check if the resolved path is within the base directory
    try:
        full_path.relative_to(base_dir)
    except ValueError as e:
        raise SecurityError(
            f"Path traversal detected: {user_path}"
        ) from e
    
    return full_path


def sanitize_for_logging(data: dict) -> dict:
    """
    Remove sensitive information from data before logging.
    
    Args:
        data: Dictionary that may contain sensitive data
        
    Returns:
        Sanitized dictionary
    """
    sensitive_keys = ['api_key', 'apikey', 'password', 'secret', 'token', 'credential']
    sanitized = data.copy()
    
    for key in list(sanitized.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = '***REDACTED***'
    
    return sanitized


def generate_unique_filename(directory: Path, filename: str) -> str:
    """
    Generate a unique filename to prevent overwrites.
    
    Args:
        directory: Directory where file will be saved
        filename: Desired filename
        
    Returns:
        Unique filename
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    
    while (directory / unique_filename).exists():
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    return unique_filename
