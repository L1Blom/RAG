"""Unit tests for DocumentLoaderRegistry."""

import pytest
from rag.services.document_loader.registry import DocumentLoaderRegistry
from rag.services.document_loader.pdf_loader import PDFLoaderStrategy
from rag.services.document_loader.text_loader import TextLoaderStrategy
from rag.services.document_loader.html_loader import HTMLLoaderStrategy
from rag.services.document_loader.word_loader import WordLoaderStrategy
from rag.services.document_loader.powerpoint_loader import PowerPointLoaderStrategy
from rag.services.document_loader.excel_loader import ExcelLoaderStrategy


def test_registry_initialization():
    """Test registry initializes with all default loaders."""
    registry = DocumentLoaderRegistry()
    types = registry.list_supported_types()
    assert set(types) == {'pdf', 'txt', 'html', 'docx', 'pptx', 'xlsx'}


def test_registry_rejects_unknown_type():
    """Test registry rejects unknown file types."""
    registry = DocumentLoaderRegistry()
    with pytest.raises(ValueError, match="Unknown file type"):
        registry.load_documents('zzz', {}, None)


def test_registry_register_custom_loader():
    """Test registering a custom loader."""
    registry = DocumentLoaderRegistry()
    
    class MarkdownLoader:
        def load(self, config, vectorstore):
            return 0
    
    registry.register_loader('md', MarkdownLoader())
    assert 'md' in registry.list_supported_types()


def test_html_loader_load_urls(temp_dir):
    """Test HTML loader URL loading from file."""
    import json
    urls = ["https://example.com", "https://test.com"]
    urls_file = temp_dir / "urls.json"
    urls_file.write_text(json.dumps(urls))
    
    loader = HTMLLoaderStrategy()
    result = loader._load_urls(str(temp_dir))
    assert result == urls


def test_html_loader_empty_urls_file(temp_dir):
    """Test HTML loader with empty URLs file."""
    urls_file = temp_dir / "urls.json"
    urls_file.write_text("")
    
    loader = HTMLLoaderStrategy()
    result = loader._load_urls(str(temp_dir))
    assert result == []


def test_html_loader_no_urls_file(temp_dir):
    """Test HTML loader with no URLs file."""
    loader = HTMLLoaderStrategy()
    result = loader._load_urls(str(temp_dir))
    assert result is None


def test_html_loader_invalid_json(temp_dir):
    """Test HTML loader with invalid JSON."""
    urls_file = temp_dir / "urls.json"
    urls_file.write_text("not valid json{{{")
    
    loader = HTMLLoaderStrategy()
    result = loader._load_urls(str(temp_dir))
    assert result is None
