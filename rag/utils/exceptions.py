"""Custom exceptions for RAG service."""


class RAGException(Exception):
    """Base exception for RAG service."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass


class ProviderError(RAGException):
    """Raised when LLM provider encounters an error."""
    pass


class DocumentLoadError(RAGException):
    """Raised when document loading fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class SecurityError(RAGException):
    """Raised when security validation fails."""
    pass
