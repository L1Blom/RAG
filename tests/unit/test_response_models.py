"""Unit tests for response models."""

import pytest
from rag.models.response_models import APIResponse, success_response, error_response


def test_api_response_success():
    """Test creating a success response."""
    response = APIResponse(success=True, data={"key": "value"})
    assert response.success is True
    assert response.data == {"key": "value"}
    assert response.error is None
    assert response.timestamp  # Should have auto-generated timestamp


def test_api_response_error():
    """Test creating an error response."""
    response = APIResponse(success=False, error="Something went wrong")
    assert response.success is False
    assert response.data is None
    assert response.error == "Something went wrong"


def test_success_response_helper():
    """Test success_response helper function."""
    result = success_response({"answer": "42"})
    assert result['success'] is True
    assert result['data'] == {"answer": "42"}
    assert result['error'] is None


def test_error_response_helper():
    """Test error_response helper function."""
    body, status_code = error_response("Bad request", 400)
    assert body['success'] is False
    assert body['error'] == "Bad request"
    assert status_code == 400


def test_error_response_default_status():
    """Test error_response default status code."""
    body, status_code = error_response("Error")
    assert status_code == 400
