"""Nebul LLM provider implementation."""

import os
import logging
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .base import LLMProvider
from rag.config.settings import ConfigService


class NebulProvider(LLMProvider):
    """Nebul provider implementation (OpenAI-compatible API)."""
    
    def __init__(self, config, config_service: Optional[ConfigService] = None):
        """
        Initialize Nebul provider.
        
        Args:
            config: RAG configuration instance
            config_service: Optional ConfigService for reading Nebul-specific settings
        """
        super().__init__(config)
        self._config_service = config_service
    
    def _get_base_url(self) -> str:
        """Get the Nebul API base URL."""
        if self._config_service:
            return self._config_service.get_string('LLMS.NEBUL', 'base_url')
        raise ValueError("ConfigService required to get Nebul base_url")
    
    def _get_api_key(self) -> str:
        """Get the Nebul API key."""
        api_key = os.environ.get('NEBUL_APIKEY')
        if not api_key:
            raise ValueError("NEBUL_APIKEY not found in environment")
        return api_key
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available Nebul models and embeddings from config."""
        if self._config_service:
            model_names = self._config_service.get_list('LLMS.NEBUL', 'models', default=[])
            embedding_names = self._config_service.get_list('LLMS.NEBUL', 'embeddings', default=[])
            return model_names, embedding_names
        
        raise ValueError("ConfigService required for Nebul model discovery")
    
    def create_chat_model(self, temperature: Optional[float] = None):
        """Create Nebul chat model via OpenAI-compatible API."""
        api_key = self._get_api_key()
        base_url = self._get_base_url()
        temp = temperature if temperature is not None else self.config.temperature
        
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=self.config.model_text,
            temperature=temp,
            verbose=True
        )
    
    def create_embeddings(self):
        """Create Nebul embeddings via OpenAI-compatible API."""
        api_key = self._get_api_key()
        base_url = self._get_base_url()
        
        return OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=self.config.embedding_model
        )
