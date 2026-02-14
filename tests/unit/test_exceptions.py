"""Unit tests for custom exceptions."""

import pytest
from rag.utils.exceptions import (
    RAGException, ConfigurationError, ProviderError,
    DocumentLoadError, VectorStoreError, ValidationError, SecurityError
)


def test_rag_exception():
    """Test base RAG exception."""
    with pytest.raises(RAGException):
        raise RAGException("base error")


def test_configuration_error():
    """Test ConfigurationError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise ConfigurationError("config error")
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("config error")


def test_provider_error():
    """Test ProviderError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise ProviderError("provider error")


def test_document_load_error():
    """Test DocumentLoadError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise DocumentLoadError("load error")


def test_vector_store_error():
    """Test VectorStoreError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise VectorStoreError("store error")


def test_validation_error():
    """Test ValidationError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise ValidationError("validation error")


def test_security_error():
    """Test SecurityError inherits from RAGException."""
    with pytest.raises(RAGException):
        raise SecurityError("security error")


def test_exception_messages():
    """Test exception messages are preserved."""
    try:
        raise ConfigurationError("specific config issue")
    except ConfigurationError as e:
        assert str(e) == "specific config issue"
