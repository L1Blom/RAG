"""Groq LLM provider implementation."""

import os
import logging
from typing import List, Tuple, Optional
from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq provider implementation."""
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available Groq models. Falls back to OpenAI for embeddings."""
        api_key = os.environ.get('GROQ_APIKEY')
        if not api_key:
            raise ValueError("GROQ_APIKEY not found in environment")
        
        client = Groq(api_key=api_key)
        models = client.models.list().data
        
        # Groq uses OpenAI-compatible model listing
        embedding_names = sorted([
            str(dict(model)['id'])
            for model in models
            if str(dict(model)['id']).startswith('text')
        ])
        
        model_names = sorted([
            str(dict(model)['id'])
            for model in models
            if str(dict(model)['id']).startswith(('gpt', 'o1', 'o3'))
            and 'audio' not in str(dict(model)['id'])
            and 'video' not in str(dict(model)['id'])
        ])
        
        return model_names, embedding_names
    
    def create_chat_model(self, temperature: Optional[float] = None):
        """Create Groq chat model."""
        api_key = os.environ.get('GROQ_APIKEY')
        if not api_key:
            raise ValueError("GROQ_APIKEY not found in environment")
        
        temp = temperature if temperature is not None else self.config.temperature
        
        return ChatGroq(
            api_key=api_key,
            model=self.config.model_text,
            temp=temp
        )
    
    def create_embeddings(self):
        """Create embeddings. Groq falls back to OpenAI embeddings."""
        api_key = os.environ.get('OPENAI_APIKEY')
        return OpenAIEmbeddings(
            api_key=api_key,
            model=self.config.embedding_model
        )
