"""Nebul LLM provider implementation."""

import os
import time
import logging
import requests
from typing import List, Tuple, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .base import LLMProvider
from rag.config.settings import ConfigService


# Cache for Nebul models with TTL
_nebul_cache: Dict[str, Any] = {
    'models': None,
    'embeddings': None,
    'timestamp': 0,
}
_CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL


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
    
    def _fetch_models_from_api(self) -> Tuple[List[str], List[str]]:
        """
        Fetch available models directly from the Nebul API.
        
        Returns:
            Tuple of (chat_models, embedding_models)
        """
        base_url = self._get_base_url().rstrip('/')
        api_key = self._get_api_key()
        
        models_url = f"{base_url}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            logging.info("Fetching live models from Nebul API: %s", models_url)
            response = requests.get(models_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            models_data = data.get('data', [])
            
            chat_models = []
            embedding_models = []
            
            for model in models_data:
                model_id = model.get('id', '')
                if not model_id:
                    continue
                
                # Categorize models based on their ID patterns
                # Embedding models typically contain 'embed', 'e5', 'minilm', etc.
                model_lower = model_id.lower()
                if any(keyword in model_lower for keyword in ['embed', 'e5', 'minilm', 'retriever']):
                    embedding_models.append(model_id)
                else:
                    chat_models.append(model_id)
            
            logging.info(
                "Nebul API returned %d chat models and %d embedding models",
                len(chat_models), len(embedding_models)
            )
            return chat_models, embedding_models
            
        except requests.exceptions.RequestException as e:
            logging.warning("Failed to fetch models from Nebul API: %s", e)
            raise
        except (KeyError, ValueError) as e:
            logging.warning("Failed to parse Nebul API response: %s", e)
            raise
    
    def _get_config_models(self) -> Tuple[List[str], List[str]]:
        """Get models from config file as fallback."""
        if self._config_service:
            model_names = self._config_service.get_list('LLMS.NEBUL', 'models', default=[])
            embedding_names = self._config_service.get_list('LLMS.NEBUL', 'embeddings', default=[])
            return model_names, embedding_names
        raise ValueError("ConfigService required for Nebul model discovery")
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """
        Get available Nebul models and embeddings.
        
        First attempts to fetch live models from the Nebul API with caching.
        Falls back to config file values if the API is unavailable.
        
        Returns:
            Tuple of (model_names, embedding_names)
        """
        global _nebul_cache
        
        current_time = time.time()
        
        # Check if cache is still valid
        if (_nebul_cache['models'] is not None and 
            _nebul_cache['embeddings'] is not None and
            current_time - _nebul_cache['timestamp'] < _CACHE_TTL_SECONDS):
            logging.debug("Using cached Nebul models")
            return _nebul_cache['models'], _nebul_cache['embeddings']
        
        # Try to fetch from API
        try:
            chat_models, embedding_models = self._fetch_models_from_api()
            
            # Update cache
            _nebul_cache['models'] = chat_models
            _nebul_cache['embeddings'] = embedding_models
            _nebul_cache['timestamp'] = current_time
            
            return chat_models, embedding_models
            
        except Exception as e:
            logging.warning("Nebul API unavailable, falling back to config: %s", e)
            
            # If we have stale cache, use it
            if _nebul_cache['models'] is not None:
                logging.info("Using stale cached Nebul models")
                return _nebul_cache['models'], _nebul_cache['embeddings']
            
            # Fall back to config file
            return self._get_config_models()
    
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
