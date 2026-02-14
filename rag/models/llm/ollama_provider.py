"""Ollama LLM provider implementation."""

import logging
import subprocess
from typing import List, Tuple, Optional
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available Ollama models by running 'ollama list'."""
        model_names = []
        
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            namelist = result.stdout.decode('utf-8').strip().split()
            for name in namelist:
                hit = name.find(':latest')
                if hit > 0:
                    model = name[0:hit]
                    model_names.append(model)
        except FileNotFoundError:
            logging.error("Ollama not found. Is it installed?")
        
        # Ollama doesn't provide embeddings; fall back to OpenAI
        embedding_names = []
        
        return model_names, embedding_names
    
    def create_chat_model(self, temperature: Optional[float] = None):
        """Create Ollama chat model."""
        return Ollama(model=self.config.model_text)
    
    def create_embeddings(self):
        """Create embeddings. Ollama falls back to OpenAI embeddings."""
        return OpenAIEmbeddings()
