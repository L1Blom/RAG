"""Ollama LLM provider implementation."""

import logging
import urllib.request
import json
from typing import List, Tuple, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from .base import LLMProvider

DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""

    def __init__(self, config, config_service=None):
        super().__init__(config)
        self._config_service = config_service

    def _get_base_url(self) -> str:
        if self._config_service:
            return self._config_service.get_string(
                'LLMS.OLLAMA', 'base_url', default=DEFAULT_OLLAMA_BASE_URL
            ).rstrip('/')
        return DEFAULT_OLLAMA_BASE_URL

    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available Ollama models via the HTTP API."""
        base_url = self._get_base_url()
        model_names = []
        try:
            req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
            data = json.loads(req.read().decode())
            for m in data.get('models', []):
                name = m.get('name', '')
                model_names.append(name.split(':')[0] if ':' in name else name)
        except Exception as e:
            logging.error("Could not reach Ollama at %s: %s", base_url, e)

        # Any Ollama model can be used for embeddings; return configured name so
        # app.py validation passes.
        embedding_names = [self.config.embedding_model] if self.config.embedding_model else model_names[:]
        return model_names, embedding_names

    def create_chat_model(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """Create Ollama chat model."""
        return ChatOllama(
            model=self.config.model_text,
            base_url=self._get_base_url(),
        )

    def create_embeddings(self):
        """Create Ollama embeddings."""
        return OllamaEmbeddings(
            model=self.config.embedding_model or self.config.model_text,
            base_url=self._get_base_url(),
        )
