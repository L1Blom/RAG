"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from rag.config.settings import ConfigService
from rag.models.config_models import RAGConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file."""
    config_content = """
[DEFAULT]
description = Test configuration
temperature = 0.7
similar = 5
score = 0.5
max_tokens = 1000
chunk_size = 1000
chunk_overlap = 200
contextualize_q_system_prompt = Test prompt
system_prompt = Test system prompt
data_dir = data/test
persistence = data/test/vectorstore
html = data/test/html
logging_level = INFO

[FLASK]
port = 8888
max_mb_size = 16

[LLMS]
use_llm = OPENAI

[LLMS.OPENAI]
modeltext = gpt-4o
embedding_model = text-embedding-3-small
"""
    config_file = temp_dir / "test_config.ini"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def config_service(sample_config_file):
    """Create a ConfigService instance."""
    return ConfigService(sample_config_file)


@pytest.fixture
def rag_config(config_service):
    """Create a RAGConfig instance."""
    return RAGConfig.from_config_file(config_service, 'test')
