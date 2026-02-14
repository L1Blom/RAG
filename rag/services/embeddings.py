"""Embeddings service for creating embedding functions."""

import logging
from langchain_core.embeddings import Embeddings
from rag.models.config_models import RAGConfig
from rag.models.llm.factory import ProviderFactory


class EmbeddingsService:
    """Service for managing embeddings."""
    
    def __init__(self, config: RAGConfig, config_service=None):
        """
        Initialize embeddings service.
        
        Args:
            config: RAG configuration instance
            config_service: Optional ConfigService for providers that need
                           INI-level settings (Azure, Nebul).
        """
        self.config = config
        self._config_service = config_service
        self._embeddings = None
    
    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings instance (lazy loaded)."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance from provider."""
        provider = ProviderFactory.get_provider(
            self.config.use_llm,
            self.config,
            config_service=self._config_service
        )
        logging.info(
            "Creating embeddings: %s with model %s",
            self.config.use_llm,
            self.config.embedding_model
        )
        return provider.create_embeddings()
    
    def reset(self) -> None:
        """Reset embeddings instance (useful for model changes)."""
        self._embeddings = None
