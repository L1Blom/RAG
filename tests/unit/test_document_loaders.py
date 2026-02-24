"""Unit tests for DocumentLoaderRegistry."""

import pytest
from rag.services.document_loader.registry import DocumentLoaderRegistry
from rag.services.document_loader.pdf_loader import PDFLoaderStrategy
from rag.services.document_loader.text_loader import TextLoaderStrategy
from rag.services.document_loader.html_loader import HTMLLoaderStrategy
from rag.services.document_loader.word_loader import WordLoaderStrategy
from rag.services.document_loader.powerpoint_loader import PowerPointLoaderStrategy
from rag.services.document_loader.excel_loader import ExcelLoaderStrategy
from rag.services.document_loader.x_loader import XLoaderStrategy


def test_registry_initialization():
    """Test registry initializes with all default loaders."""
    registry = DocumentLoaderRegistry()
    types = registry.list_supported_types()
    assert set(types) == {'pdf', 'txt', 'html', 'docx', 'pptx', 'xlsx', 'x'}


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


# ---------------------------------------------------------------------------
# X Loader Tests
# ---------------------------------------------------------------------------

def test_x_loader_extract_tweet_id():
    """Test extracting tweet ID from various X URL formats."""
    loader = XLoaderStrategy()
    
    # Standard x.com URL
    assert loader._extract_tweet_id("https://x.com/user/status/1234567890") == "1234567890"
    # www.x.com URL
    assert loader._extract_tweet_id("https://www.x.com/user/status/1234567890") == "1234567890"
    # twitter.com URL
    assert loader._extract_tweet_id("https://twitter.com/user/status/1234567890") == "1234567890"
    # With trailing slash
    assert loader._extract_tweet_id("https://x.com/user/status/1234567890/") == "1234567890"
    # Non-matching URL
    assert loader._extract_tweet_id("https://example.com/post/123") is None


def test_x_loader_is_x_url():
    """Test X URL validation."""
    loader = XLoaderStrategy()
    
    # Valid X URLs
    assert loader._is_x_url("https://x.com/user/status/1234567890")
    assert loader._is_x_url("https://twitter.com/user/status/1234567890")
    assert loader._is_x_url("http://www.x.com/user/status/1234567890")
    # Invalid URLs
    assert not loader._is_x_url("https://example.com/user/status/1234567890")
    assert not loader._is_x_url("https://facebook.com/user/posts/123")
    assert not loader._is_x_url("not a url")


def test_x_loader_tweet_to_document():
    """Test converting tweet data to LangChain Document."""
    loader = XLoaderStrategy()
    
    tweet_data = {
        "data": {
            "id": "1234567890",
            "author_id": "987654321",
            "text": "Hello from X!",
            "created_at": "2024-01-15T10:30:00.000Z",
            "public_metrics": {
                "retweet_count": 10,
                "like_count": 50,
                "reply_count": 5,
                "quote_count": 2
            }
        },
        "includes": {
            "users": [{
                "id": "987654321",
                "username": "testuser",
                "name": "Test User"
            }]
        }
    }
    
    doc = loader._tweet_to_document(tweet_data)
    
    assert doc.page_content is not None
    assert "testuser" in doc.page_content
    assert "Test User" in doc.page_content
    assert "Hello from X!" in doc.page_content
    assert doc.metadata["tweet_id"] == "1234567890"
    assert doc.metadata["author_username"] == "testuser"
    assert doc.metadata["like_count"] == 50


def test_x_loader_api_token_from_env(monkeypatch):
    """Test X loader gets API token from environment."""
    # Clear any existing token
    monkeypatch.delenv('X_API_KEY', raising=False)
    monkeypatch.delenv('TWITTER_BEARER_TOKEN', raising=False)
    
    loader = XLoaderStrategy()
    assert loader.api_token is None
    
    # Set environment variable
    monkeypatch.setenv('X_API_KEY', 'test_bearer_token')
    
    loader2 = XLoaderStrategy()
    assert loader2.api_token == 'test_bearer_token'


def test_x_loader_load_no_urls():
    """Test X loader with no URLs provided."""
    loader = XLoaderStrategy()
    
    # Should return 0 when no URLs provided
    result = loader.load({}, None)
    assert result == 0
