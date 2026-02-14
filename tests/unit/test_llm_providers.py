"""Unit tests for LLM providers and factory."""

import pytest
from rag.models.llm.base import LLMProvider
from rag.models.llm.factory import ProviderFactory
from rag.models.llm.openai_provider import OpenAIProvider
from rag.models.llm.azure_provider import AzureProvider
from rag.models.llm.groq_provider import GroqProvider
from rag.models.llm.ollama_provider import OllamaProvider
from rag.models.llm.nebul_provider import NebulProvider
from rag.config.settings import ConfigService


def test_factory_lists_all_providers():
    """Test that factory lists all expected providers."""
    providers = ProviderFactory.list_providers()
    assert 'OPENAI' in providers
    assert 'AZURE' in providers
    assert 'GROQ' in providers
    assert 'OLLAMA' in providers
    assert 'NEBUL' in providers


def test_factory_rejects_unknown_provider(rag_config):
    """Test that factory rejects unknown provider names."""
    with pytest.raises(ValueError, match="Unknown provider"):
        ProviderFactory.get_provider('NONEXISTENT', rag_config)


def test_factory_creates_openai_provider(rag_config):
    """Test that factory creates OpenAI provider."""
    provider = ProviderFactory.get_provider('OPENAI', rag_config)
    assert isinstance(provider, OpenAIProvider)


def test_factory_creates_azure_provider(rag_config):
    """Test that factory creates Azure provider."""
    provider = ProviderFactory.get_provider('AZURE', rag_config)
    assert isinstance(provider, AzureProvider)


def test_factory_creates_groq_provider(rag_config):
    """Test that factory creates Groq provider."""
    provider = ProviderFactory.get_provider('GROQ', rag_config)
    assert isinstance(provider, GroqProvider)


def test_factory_creates_ollama_provider(rag_config):
    """Test that factory creates Ollama provider."""
    provider = ProviderFactory.get_provider('OLLAMA', rag_config)
    assert isinstance(provider, OllamaProvider)


def test_factory_creates_nebul_provider(rag_config):
    """Test that factory creates Nebul provider."""
    provider = ProviderFactory.get_provider('NEBUL', rag_config)
    assert isinstance(provider, NebulProvider)


def test_factory_register_custom_provider(rag_config):
    """Test registering a custom provider."""
    class CustomProvider(LLMProvider):
        def get_model_names(self): return [], []
        def create_chat_model(self, temperature=None): pass
        def create_embeddings(self): pass

    ProviderFactory.register_provider('CUSTOM', CustomProvider)
    provider = ProviderFactory.get_provider('CUSTOM', rag_config)
    assert isinstance(provider, CustomProvider)
    # Clean up
    del ProviderFactory._providers['CUSTOM']


def test_nebul_provider_with_config_service(sample_config_file, temp_dir):
    """Test NebulProvider with ConfigService providing model lists."""
    # Add NEBUL sections to config
    config_content = (temp_dir / "test_config.ini").read_text()
    config_content += """
[LLMS.NEBUL]
models = Qwen/Qwen3-30B-A3B-Instruct-2507,meta-llama/Llama-4-Maverick-17B-128E-Instruct
embeddings = Qwen/Qwen3-Embedding-8B,intfloat/multilingual-e5-large-instruct
base_url = https://api.inference.nebul.io/v1
modeltext = Qwen/Qwen3-30B-A3B-Instruct-2507
embedding_model = Qwen/Qwen3-Embedding-8B
"""
    (temp_dir / "test_config.ini").write_text(config_content)
    cs = ConfigService(str(temp_dir / "test_config.ini"))
    
    from rag.models.config_models import RAGConfig
    rag_config = RAGConfig.from_config_file(cs, 'test')
    
    provider = NebulProvider(rag_config, config_service=cs)
    models, embeddings = provider.get_model_names()
    assert len(models) == 2
    assert len(embeddings) == 2


def test_nebul_provider_without_config_service(rag_config):
    """Test NebulProvider raises without config service."""
    provider = NebulProvider(rag_config)
    with pytest.raises(ValueError, match="ConfigService required"):
        provider.get_model_names()


def test_provider_base_class_is_abstract():
    """Test that LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider(None)
