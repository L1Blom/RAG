"""Unit tests for ConfigService."""

import pytest
from rag.config.settings import ConfigService, ConfigError


def test_config_service_initialization(sample_config_file):
    """Test ConfigService initialization."""
    config = ConfigService(sample_config_file)
    assert config.config_file.exists()


def test_config_service_missing_file():
    """Test ConfigService with missing file."""
    with pytest.raises(ConfigError):
        ConfigService('/nonexistent/path/config.ini')


def test_get_string(config_service):
    """Test getting string values."""
    value = config_service.get_string('DEFAULT', 'description')
    assert value == 'Test configuration'


def test_get_int(config_service):
    """Test getting integer values."""
    value = config_service.get_int('FLASK', 'port')
    assert value == 8888


def test_get_float(config_service):
    """Test getting float values."""
    value = config_service.get_float('DEFAULT', 'temperature')
    assert value == 0.7


def test_get_boolean(temp_dir):
    """Test getting boolean values."""
    config_content = """
[DEFAULT]
debug = False
enabled = True

[TEST]
verbose = Yes
"""
    config_file = temp_dir / "bool_config.ini"
    config_file.write_text(config_content)
    cs = ConfigService(str(config_file))
    assert cs.get_boolean('DEFAULT', 'debug') is False
    assert cs.get_boolean('DEFAULT', 'enabled') is True
    assert cs.get_boolean('TEST', 'verbose') is True


def test_get_list(config_service):
    """Test getting list values."""
    value = config_service.get_list('LLMS', 'use_llm')
    assert value == ['OPENAI']


def test_get_with_default(config_service):
    """Test getting value with default."""
    value = config_service.get_string('NONEXISTENT', 'key', default='default')
    assert value == 'default'


def test_get_int_with_default(config_service):
    """Test getting int value with default."""
    value = config_service.get_int('NONEXISTENT', 'key', default=42)
    assert value == 42


def test_get_float_with_default(config_service):
    """Test getting float value with default."""
    value = config_service.get_float('NONEXISTENT', 'key', default=3.14)
    assert value == 3.14


def test_missing_key_raises_error(config_service):
    """Test that missing key raises ConfigError."""
    with pytest.raises(ConfigError):
        config_service.get_string('NONEXISTENT', 'key')


def test_missing_int_raises_error(config_service):
    """Test that missing int key raises ConfigError."""
    with pytest.raises(ConfigError):
        config_service.get_int('NONEXISTENT', 'key')


def test_missing_float_raises_error(config_service):
    """Test that missing float key raises ConfigError."""
    with pytest.raises(ConfigError):
        config_service.get_float('NONEXISTENT', 'key')


def test_has_option(config_service):
    """Test checking if option exists."""
    assert config_service.has_option('DEFAULT', 'description')
    assert not config_service.has_option('DEFAULT', 'nonexistent')
