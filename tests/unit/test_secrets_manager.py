"""Unit tests for SecretsManager."""

import os
import pytest
from pathlib import Path
from rag.config.secrets import SecretsManager


def test_secrets_manager_initialization(temp_dir):
    """Test SecretsManager initialization."""
    manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
    assert manager.secrets_dir == temp_dir


def test_get_secret_from_env(temp_dir):
    """Test getting secret from environment variable."""
    os.environ['TEST_SECRET_KEY'] = 'test_value'
    try:
        manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
        assert manager.get_secret('TEST_SECRET_KEY') == 'test_value'
    finally:
        del os.environ['TEST_SECRET_KEY']


def test_get_secret_from_file(temp_dir):
    """Test getting secret from Docker secrets file."""
    secret_file = temp_dir / 'test_docker_secret'
    secret_file.write_text('docker_secret_value\n')
    
    manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
    assert manager.get_secret('TEST_DOCKER_SECRET') == 'docker_secret_value'


def test_get_secret_file_takes_priority(temp_dir):
    """Test that Docker secret file takes priority over env var."""
    secret_file = temp_dir / 'priority_key'
    secret_file.write_text('file_value')
    os.environ['PRIORITY_KEY'] = 'env_value'
    
    try:
        manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
        assert manager.get_secret('PRIORITY_KEY') == 'file_value'
    finally:
        del os.environ['PRIORITY_KEY']


def test_get_secret_not_found(temp_dir):
    """Test getting a nonexistent secret returns None."""
    manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
    assert manager.get_secret('COMPLETELY_NONEXISTENT_KEY_12345') is None


def test_load_all_secrets(temp_dir):
    """Test loading all known secrets."""
    # Create a test secret file
    secret_file = temp_dir / 'openai_apikey'
    secret_file.write_text('test_openai_key')
    
    manager = SecretsManager(secrets_dir=str(temp_dir), env_file='nonexistent.env')
    manager.load_all_secrets()
    
    assert os.environ.get('OPENAI_APIKEY') == 'test_openai_key'
    # Clean up
    if 'OPENAI_APIKEY' in os.environ:
        del os.environ['OPENAI_APIKEY']
