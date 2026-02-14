"""Unit tests for security utilities."""

import pytest
from pathlib import Path
from rag.utils.security import (
    allowed_file, secure_upload_filename, validate_path,
    sanitize_for_logging, generate_unique_filename
)
from rag.utils.exceptions import SecurityError, ValidationError


def test_allowed_file_valid():
    """Test allowed file extensions."""
    assert allowed_file('document.pdf') is True
    assert allowed_file('notes.txt') is True
    assert allowed_file('report.docx') is True
    assert allowed_file('slides.pptx') is True
    assert allowed_file('data.xlsx') is True
    assert allowed_file('page.html') is True


def test_allowed_file_invalid():
    """Test rejected file extensions."""
    assert allowed_file('script.exe') is False
    assert allowed_file('hack.sh') is False
    assert allowed_file('noextension') is False
    assert allowed_file('') is False


def test_secure_upload_filename_valid():
    """Test securing a valid filename."""
    result = secure_upload_filename('document.pdf')
    assert result == 'document.pdf'


def test_secure_upload_filename_invalid():
    """Test securing an invalid filename."""
    with pytest.raises(ValidationError):
        secure_upload_filename('script.exe')


def test_secure_upload_filename_empty():
    """Test securing an empty filename."""
    with pytest.raises(ValidationError):
        secure_upload_filename('')


def test_validate_path_safe(temp_dir):
    """Test safe path validation."""
    result = validate_path(temp_dir, 'subdir/file.txt')
    assert str(result).startswith(str(temp_dir.resolve()))


def test_validate_path_traversal(temp_dir):
    """Test path traversal detection."""
    with pytest.raises(SecurityError, match="Path traversal detected"):
        validate_path(temp_dir, '../../etc/passwd')


def test_sanitize_for_logging():
    """Test sensitive data sanitization."""
    data = {
        'name': 'test',
        'api_key': 'secret123',
        'OPENAI_APIKEY': 'sk-xxx',
        'password': 'pass123',
        'port': 8080
    }
    result = sanitize_for_logging(data)
    assert result['name'] == 'test'
    assert result['api_key'] == '***REDACTED***'
    assert result['OPENAI_APIKEY'] == '***REDACTED***'
    assert result['password'] == '***REDACTED***'
    assert result['port'] == 8080


def test_generate_unique_filename(temp_dir):
    """Test unique filename generation."""
    # No conflict
    result = generate_unique_filename(temp_dir, 'file.txt')
    assert result == 'file.txt'

    # Create a conflicting file
    (temp_dir / 'file.txt').write_text('exists')
    result = generate_unique_filename(temp_dir, 'file.txt')
    assert result == 'file_1.txt'

    # Create another conflict
    (temp_dir / 'file_1.txt').write_text('exists')
    result = generate_unique_filename(temp_dir, 'file.txt')
    assert result == 'file_2.txt'
