"""OpenAI LLM provider implementation."""

import os
import logging
from typing import List, Tuple, Optional
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available OpenAI models and embeddings."""
        api_key = os.environ.get('OPENAI_APIKEY')
        if not api_key:
            raise ValueError("OPENAI_APIKEY not found in environment")
        
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list().data
        
        # Filter embedding models
        embedding_names = sorted([
            str(dict(model)['id'])
            for model in models
            if str(dict(model)['id']).startswith('text')
        ])
        
        # Filter chat models
        model_names = sorted([
            str(dict(model)['id'])
            for model in models
            if str(dict(model)['id']).startswith(('gpt', 'o1', 'o3'))
            and 'audio' not in str(dict(model)['id'])
            and 'video' not in str(dict(model)['id'])
        ])
        
        return model_names, embedding_names
    
    def create_chat_model(self, temperature: Optional[float] = None):
        """Create OpenAI chat model."""
        api_key = os.environ.get('OPENAI_APIKEY')
        if not api_key:
            raise ValueError("OPENAI_APIKEY not found in environment")
        
        temp = temperature if temperature is not None else self.config.temperature
        
        return ChatOpenAI(
            api_key=api_key,
            model=self.config.model_text,
            temperature=temp,
            verbose=True
        )
    
    def create_embeddings(self):
        """Create OpenAI embeddings."""
        api_key = os.environ.get('OPENAI_APIKEY')
        if not api_key:
            raise ValueError("OPENAI_APIKEY not found in environment")
        
        return OpenAIEmbeddings(
            api_key=api_key,
            model=self.config.embedding_model
        )
