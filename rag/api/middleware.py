"""Flask middleware and error handlers."""

import logging
from flask import Flask, Response, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from rag.models.response_models import error_response


def setup_middleware(app: Flask) -> None:
    """
    Setup middleware for Flask app.
    
    Configures:
        - CORS support
        - Preflight request handling
        - HTTP error handler
        - Unhandled exception handler
        - Teardown request logging
    
    Args:
        app: Flask application instance
    """
    # Enable CORS
    CORS(app)

    @app.before_request
    def handle_preflight():
        """Handle CORS preflight OPTIONS requests."""
        if request.method == "OPTIONS":
            res = Response()
            res.headers['X-Content-Type-Options'] = '*'
            return res

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle HTTP exceptions with standard error response."""
        logging.error("HTTP error: %s", e)
        return error_response(str(e), e.code)

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle unexpected exceptions."""
        logging.error("Unexpected error: %s", e, exc_info=True)
        return error_response("Internal server error", 500)

    @app.teardown_request
    def log_unhandled(e):
        """Log any unhandled exceptions during teardown."""
        if e is not None:
            logging.error("Unhandled exception during request: %s", repr(e))
