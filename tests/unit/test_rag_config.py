"""Unit tests for RAGConfig."""

import pytest
from rag.models.config_models import RAGConfig
from rag.config.settings import ConfigService


def test_rag_config_from_config_file(config_service):
    """Test creating RAGConfig from ConfigService."""
    config = RAGConfig.from_config_file(config_service, 'test')
    assert config.project == 'test'
    assert config.use_llm == 'OPENAI'
    assert config.model_text == 'gpt-4o'
    assert config.embedding_model == 'text-embedding-3-small'
    assert config.temperature == 0.7
    assert config.port == 8888


def test_rag_config_temperature_validation():
    """Test temperature validation."""
    with pytest.raises(ValueError, match="Temperature must be between"):
        RAGConfig(
            project='test', use_llm='TEST', model_text='m',
            embedding_model='e', temperature=3.0, max_tokens=100,
            similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
            contextualize_q_system_prompt='q', system_prompt='s',
            data_dir='d', persistence_dir='p', html_dir='h',
            port=5000, max_mb_size=16, logging_level='INFO'
        )


def test_rag_config_negative_temperature():
    """Test negative temperature is rejected."""
    with pytest.raises(ValueError, match="Temperature must be between"):
        RAGConfig(
            project='test', use_llm='TEST', model_text='m',
            embedding_model='e', temperature=-0.5, max_tokens=100,
            similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
            contextualize_q_system_prompt='q', system_prompt='s',
            data_dir='d', persistence_dir='p', html_dir='h',
            port=5000, max_mb_size=16, logging_level='INFO'
        )


def test_rag_config_empty_data_dir():
    """Test empty data_dir is rejected."""
    with pytest.raises(ValueError, match="data_dir must be specified"):
        RAGConfig(
            project='test', use_llm='TEST', model_text='m',
            embedding_model='e', temperature=0.5, max_tokens=100,
            similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
            contextualize_q_system_prompt='q', system_prompt='s',
            data_dir='', persistence_dir='p', html_dir='h',
            port=5000, max_mb_size=16, logging_level='INFO'
        )


def test_rag_config_empty_persistence_dir():
    """Test empty persistence_dir is rejected."""
    with pytest.raises(ValueError, match="persistence_dir must be specified"):
        RAGConfig(
            project='test', use_llm='TEST', model_text='m',
            embedding_model='e', temperature=0.5, max_tokens=100,
            similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
            contextualize_q_system_prompt='q', system_prompt='s',
            data_dir='d', persistence_dir='', html_dir='h',
            port=5000, max_mb_size=16, logging_level='INFO'
        )


def test_rag_config_valid_temperature_bounds():
    """Test valid temperature bounds (0.0 and 2.0)."""
    config_min = RAGConfig(
        project='test', use_llm='TEST', model_text='m',
        embedding_model='e', temperature=0.0, max_tokens=100,
        similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
        contextualize_q_system_prompt='q', system_prompt='s',
        data_dir='d', persistence_dir='p', html_dir='h',
        port=5000, max_mb_size=16, logging_level='INFO'
    )
    assert config_min.temperature == 0.0

    config_max = RAGConfig(
        project='test', use_llm='TEST', model_text='m',
        embedding_model='e', temperature=2.0, max_tokens=100,
        similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
        contextualize_q_system_prompt='q', system_prompt='s',
        data_dir='d', persistence_dir='p', html_dir='h',
        port=5000, max_mb_size=16, logging_level='INFO'
    )
    assert config_max.temperature == 2.0


def test_rag_config_optional_fields():
    """Test optional fields default to None."""
    config = RAGConfig(
        project='test', use_llm='TEST', model_text='m',
        embedding_model='e', temperature=0.5, max_tokens=100,
        similar=4, score=0.1, chunk_size=80, chunk_overlap=10,
        contextualize_q_system_prompt='q', system_prompt='s',
        data_dir='d', persistence_dir='p', html_dir='h',
        port=5000, max_mb_size=16, logging_level='INFO'
    )
    assert config.session_id is None
    assert config.timestamp is None
