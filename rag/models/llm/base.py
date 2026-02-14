"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from rag.models.config_models import RAGConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: RAG configuration instance
        """
        self.config = config
    
    @abstractmethod
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """
        Get available model and embedding names for this provider.
        
        Returns:
            Tuple of (model_names, embedding_names)
        """
        pass
    
    @abstractmethod
    def create_chat_model(self, temperature: Optional[float] = None):
        """
        Create a chat model instance.
        
        Args:
            temperature: Optional temperature override
            
        Returns:
            LangChain chat model instance
        """
        pass
    
    @abstractmethod
    def create_embeddings(self):
        """
        Create an embeddings instance.
        
        Returns:
            LangChain embeddings instance
        """
        pass
    
    def validate_model(self, model_name: str, embedding_name: str) -> None:
        """
        Validate that the specified models are available.
        
        Args:
            model_name: Name of the chat model
            embedding_name: Name of the embedding model
            
        Raises:
            ValueError: If models are not available
        """
        model_names, embedding_names = self.get_model_names()
        
        if model_name not in model_names:
            raise ValueError(
                f"Model '{model_name}' not found in {self.__class__.__name__}. "
                f"Available models: {', '.join(model_names)}"
            )
        
        if embedding_name not in embedding_names:
            raise ValueError(
                f"Embedding '{embedding_name}' not found in {self.__class__.__name__}. "
                f"Available embeddings: {', '.join(embedding_names)}"
            )
