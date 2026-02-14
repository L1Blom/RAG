"""Logging configuration utilities."""

import logging
from typing import Optional


def setup_logging(level: str = 'INFO') -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress werkzeug logs unless ERROR
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('chromadb').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
