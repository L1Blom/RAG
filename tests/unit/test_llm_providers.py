"""Unit tests for LLM providers and factory."""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from rag.models.llm.base import LLMProvider
from rag.models.llm.factory import ProviderFactory
from rag.models.llm.openai_provider import OpenAIProvider
from rag.models.llm.azure_provider import AzureProvider
from rag.models.llm.groq_provider import GroqProvider
from rag.models.llm.ollama_provider import OllamaProvider
from rag.models.llm.nebul_provider import NebulProvider, _nebul_cache
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


def test_nebul_provider_fetches_from_api(sample_config_file, temp_dir):
    """Test NebulProvider fetches models from live API."""
    # Setup config
    config_content = (temp_dir / "test_config.ini").read_text()
    config_content += """
[LLMS.NEBUL]
models = fallback-model
embeddings = fallback-embedding
base_url = https://api.inference.nebul.io/v1
modeltext = Qwen/Qwen3-30B-A3B-Instruct-2507
embedding_model = Qwen/Qwen3-Embedding-8B
"""
    (temp_dir / "test_config.ini").write_text(config_content)
    cs = ConfigService(str(temp_dir / "test_config.ini"))
    
    from rag.models.config_models import RAGConfig
    rag_config = RAGConfig.from_config_file(cs, 'test')
    
    provider = NebulProvider(rag_config, config_service=cs)
    
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'data': [
            {'id': 'Qwen/Qwen3-30B-A3B-Instruct-2507'},
            {'id': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'},
            {'id': 'Qwen/Qwen3-Embedding-8B'},
            {'id': 'intfloat/multilingual-e5-large-instruct'},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    # Clear cache before test
    import rag.models.llm.nebul_provider as nebul_module
    nebul_module._nebul_cache['models'] = None
    nebul_module._nebul_cache['embeddings'] = None
    nebul_module._nebul_cache['timestamp'] = 0
    
    with patch('requests.get', return_value=mock_response):
        with patch.dict(os.environ, {'NEBUL_APIKEY': 'test-key'}):
            models, embeddings = provider.get_model_names()
    
    # Chat models should not contain embedding-like models
    assert 'Qwen/Qwen3-30B-A3B-Instruct-2507' in models
    assert 'meta-llama/Llama-4-Maverick-17B-128E-Instruct' in models
    # Embedding models should be separated
    assert 'Qwen/Qwen3-Embedding-8B' in embeddings
    assert 'intfloat/multilingual-e5-large-instruct' in embeddings


def test_nebul_provider_fallback_to_config_on_api_error(sample_config_file, temp_dir):
    """Test NebulProvider falls back to config when API fails."""
    # Setup config with fallback values
    config_content = (temp_dir / "test_config.ini").read_text()
    config_content += """
[LLMS.NEBUL]
models = fallback-model-1,fallback-model-2
embeddings = fallback-embedding-1
base_url = https://api.inference.nebul.io/v1
modeltext = fallback-model-1
embedding_model = fallback-embedding-1
"""
    (temp_dir / "test_config.ini").write_text(config_content)
    cs = ConfigService(str(temp_dir / "test_config.ini"))
    
    from rag.models.config_models import RAGConfig
    rag_config = RAGConfig.from_config_file(cs, 'test')
    
    provider = NebulProvider(rag_config, config_service=cs)
    
    # Clear cache before test
    import rag.models.llm.nebul_provider as nebul_module
    nebul_module._nebul_cache['models'] = None
    nebul_module._nebul_cache['embeddings'] = None
    nebul_module._nebul_cache['timestamp'] = 0
    
    # Mock API failure
    with patch('requests.get', side_effect=Exception("API unavailable")):
        with patch.dict(os.environ, {'NEBUL_APIKEY': 'test-key'}):
            models, embeddings = provider.get_model_names()
    
    # Should return config fallback values
    assert models == ['fallback-model-1', 'fallback-model-2']
    assert embeddings == ['fallback-embedding-1']


def test_nebul_provider_uses_cache(sample_config_file, temp_dir):
    """Test NebulProvider uses cached models when cache is valid."""
    # Setup config
    config_content = (temp_dir / "test_config.ini").read_text()
    config_content += """
[LLMS.NEBUL]
models = fallback-model
embeddings = fallback-embedding
base_url = https://api.inference.nebul.io/v1
modeltext = Qwen/Qwen3-30B-A3B-Instruct-2507
embedding_model = Qwen/Qwen3-Embedding-8B
"""
    (temp_dir / "test_config.ini").write_text(config_content)
    cs = ConfigService(str(temp_dir / "test_config.ini"))
    
    from rag.models.config_models import RAGConfig
    rag_config = RAGConfig.from_config_file(cs, 'test')
    
    provider = NebulProvider(rag_config, config_service=cs)
    
    # Set up cache with valid data
    import rag.models.llm.nebul_provider as nebul_module
    nebul_module._nebul_cache['models'] = ['cached-model']
    nebul_module._nebul_cache['embeddings'] = ['cached-embedding']
    nebul_module._nebul_cache['timestamp'] = time.time()
    
    with patch.dict(os.environ, {'NEBUL_APIKEY': 'test-key'}):
        models, embeddings = provider.get_model_names()
    
    # Should return cached values without calling API
    assert models == ['cached-model']
    assert embeddings == ['cached-embedding']


def test_nebul_provider_uses_stale_cache_on_api_error(sample_config_file, temp_dir):
    """Test NebulProvider uses stale cache when API fails and no fresh cache."""
    # Setup config
    config_content = (temp_dir / "test_config.ini").read_text()
    config_content += """
[LLMS.NEBUL]
models = config-fallback-model
embeddings = config-fallback-embedding
base_url = https://api.inference.nebul.io/v1
modeltext = Qwen/Qwen3-30B-A3B-Instruct-2507
embedding_model = Qwen/Qwen3-Embedding-8B
"""
    (temp_dir / "test_config.ini").write_text(config_content)
    cs = ConfigService(str(temp_dir / "test_config.ini"))
    
    from rag.models.config_models import RAGConfig
    rag_config = RAGConfig.from_config_file(cs, 'test')
    
    provider = NebulProvider(rag_config, config_service=cs)
    
    # Set up stale cache (expired TTL)
    import rag.models.llm.nebul_provider as nebul_module
    nebul_module._nebul_cache['models'] = ['stale-cached-model']
    nebul_module._nebul_cache['embeddings'] = ['stale-cached-embedding']
    nebul_module._nebul_cache['timestamp'] = time.time() - 7200  # 2 hours ago
    
    with patch('requests.get', side_effect=Exception("API unavailable")):
        with patch.dict(os.environ, {'NEBUL_APIKEY': 'test-key'}):
            models, embeddings = provider.get_model_names()
    
    # Should use stale cache rather than config fallback
    assert models == ['stale-cached-model']
    assert embeddings == ['stale-cached-embedding']


def test_provider_base_class_is_abstract():
    """Test that LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider(None)
