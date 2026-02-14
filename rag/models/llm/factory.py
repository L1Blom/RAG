"""Factory for creating LLM providers."""

from typing import Dict, Optional, Type
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .azure_provider import AzureProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .nebul_provider import NebulProvider
from rag.models.config_models import RAGConfig


class ProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        'OPENAI': OpenAIProvider,
        'AZURE': AzureProvider,
        'GROQ': GroqProvider,
        'OLLAMA': OllamaProvider,
        'NEBUL': NebulProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str, config: RAGConfig,
                     config_service=None) -> LLMProvider:
        """
        Get a provider instance by name.
        
        Args:
            provider_name: Name of the provider (e.g., 'OPENAI', 'AZURE')
            config: RAG configuration instance
            config_service: Optional ConfigService for providers that need
                           INI-level settings (Azure, Nebul).
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider name is not recognized
            
        Example:
            >>> config = RAGConfig.from_config_file(config_service, 'myproject')
            >>> provider = ProviderFactory.get_provider('OPENAI', config)
            >>> chat_model = provider.create_chat_model()
        """
        provider_class = cls._providers.get(provider_name)
        if not provider_class:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {available}"
            )
        
        # Pass config_service to providers that accept it
        if config_service is not None and hasattr(provider_class.__init__, '__code__') \
                and 'config_service' in provider_class.__init__.__code__.co_varnames:
            return provider_class(config, config_service=config_service)
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a new provider class.
        
        Args:
            name: Provider name
            provider_class: Provider class to register
        """
        cls._providers[name] = provider_class
    
    @classmethod
    def list_providers(cls) -> list:
        """Get list of available provider names."""
        return list(cls._providers.keys())
