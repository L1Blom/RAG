"""Unit tests for the Flask app factory and middleware."""

import io
import pytest
from flask import Flask
from rag.api.middleware import setup_middleware
from rag.api.routes import rag_bp


def test_middleware_sets_up_cors():
    """Test that setup_middleware configures CORS without error."""
    app = Flask(__name__)
    setup_middleware(app)
    # After setup, preflight handler should be registered
    assert app.before_request_funcs is not None


def test_middleware_error_handlers():
    """Test that error handlers are registered."""
    app = Flask(__name__)
    setup_middleware(app)
    # Flask stores error handlers in a dict keyed by blueprint (None = app-level)
    assert None in app.error_handler_spec
    assert 500 in app.error_handler_spec[None] or Exception in app.error_handler_spec[None].get(None, {})


def test_blueprint_registration():
    """Test that the rag blueprint registers without error."""
    app = Flask(__name__)
    app.register_blueprint(rag_bp)
    # Check some expected routes exist
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    assert '/ping' in rules
    assert '/prompt/<project>' in rules
    assert '/prompt/<project>/search' in rules
    assert '/prompt/<project>/reload' in rules


def test_ping_route():
    """Test the /ping health check endpoint via test client."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    # Provide minimal state for the ping route
    app.config['APP_STATE'] = {
        'timestamp': '2026-02-13 12:00:00',
    }

    class _FakeConfig:
        model_text = 'test-model'

    app.config['RAG_CONFIG'] = _FakeConfig()

    with app.test_client() as client:
        resp = client.get('/ping')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['answer']['timestamp'] == '2026-02-13 12:00:00'
        assert data['answer']['llm'] == 'test-model'


def test_cache_module():
    """Test the cache utility module."""
    from rag.utils.cache import cached_config, clear_all_caches
    import tempfile, os

    # Create a small INI file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("[section1]\nkey1 = value1\n[section2]\nkey2 = value2\n")
        path = f.name

    try:
        result = cached_config(path)
        assert 'section1' in result
        assert result['section1']['key1'] == 'value1'
        assert result['section2']['key2'] == 'value2'

        # Second call should hit cache (same object identity)
        result2 = cached_config(path)
        assert result is result2

        clear_all_caches()
    finally:
        os.unlink(path)


def test_uploadx_route_stores_url_and_invokes_loader(temp_dir):
    """Test /uploadx stores URL to x.json and calls load_x_urls once."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def __init__(self):
            self.calls = []

        def load_x_urls(self, urls):
            self.calls.append(urls)
            return 1

    fake_vector = _FakeVectorStoreService()
    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = fake_vector

    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx', data={'url': 'x.com/user/status/123456'})

    assert response.status_code == 200
    assert fake_vector.calls == [['https://x.com/user/status/123456']]

    x_urls_file = temp_dir / 'x.json'
    assert x_urls_file.exists()
    assert 'https://x.com/user/status/123456' in x_urls_file.read_text()


def test_uploadx_batch_processes_valid_urls_and_updates_x_json(temp_dir):
    """Test /uploadx/batch processes valid URLs and skips invalid entries."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def __init__(self):
            self.calls = []

        def load_x_urls(self, urls):
            self.calls.append(urls)
            return 1

    fake_vector = _FakeVectorStoreService()
    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = fake_vector

    payload = "x.com/user1/status/111\nnot-a-valid-url\nhttps://twitter.com/user2/status/222\n"
    data = {'file': (io.BytesIO(payload.encode('utf-8')), 'urls.txt')}

    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    assert len(fake_vector.calls) == 2
    assert fake_vector.calls[0] == ['https://x.com/user1/status/111']
    assert fake_vector.calls[1] == ['https://twitter.com/user2/status/222']

    x_urls_file = temp_dir / 'x.json'
    contents = x_urls_file.read_text()
    assert 'https://x.com/user1/status/111' in contents
    assert 'https://twitter.com/user2/status/222' in contents


def test_uploadx_route_requires_url(temp_dir):
    """Test /uploadx returns 400 when URL parameter is missing."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 1

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx', data={})

    assert response.status_code == 400
    assert b'URL parameter is required' in response.data


def test_uploadx_route_rejects_invalid_x_url(temp_dir):
    """Test /uploadx rejects non-X URLs."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 1

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx', data={'url': 'https://example.com/post/123'})

    assert response.status_code == 400
    assert b'Invalid X URL format' in response.data


def test_uploadx_batch_requires_file_part(temp_dir):
    """Test /uploadx/batch returns 400 when no file part is present."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 1

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data={}, content_type='multipart/form-data')

    assert response.status_code == 400
    assert b'No file part in the request' in response.data


def test_uploadx_batch_rejects_empty_url_list(temp_dir):
    """Test /uploadx/batch returns 400 when uploaded file has no URLs."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 1

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    data = {'file': (io.BytesIO(b'\n\n'), 'urls.txt')}
    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    assert b'No URLs found in file' in response.data


def test_uploadx_batch_rejects_file_without_valid_x_urls(temp_dir):
    """Test /uploadx/batch returns 400 when file has no valid X URLs."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 1

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    data = {'file': (io.BytesIO(b'https://example.com/a\nhttps://foo.bar/b\n'), 'urls.txt')}
    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data=data, content_type='multipart/form-data')

    assert response.status_code == 400
    assert b'No valid X URLs found in file' in response.data


def test_uploadx_batch_single_url_processing_failure_returns_500(temp_dir):
    """Single valid URL with zero chunks should return 500 with a clear message."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            return 0

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    data = {'file': (io.BytesIO(b'https://x.com/user/status/123\n'), 'urls.txt')}
    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data=data, content_type='multipart/form-data')

    assert response.status_code == 500
    assert b'failed to process' in response.data.lower()


def test_uploadx_batch_single_url_exception_returns_500(temp_dir):
    """Single valid URL raising loader exception should return 500 with error details."""
    app = Flask(__name__)
    setup_middleware(app)
    app.register_blueprint(rag_bp)

    class _FakeConfig:
        data_dir = str(temp_dir)

    class _FakeVectorStoreService:
        def load_x_urls(self, urls):
            raise RuntimeError('boom')

    app.config['RAG_CONFIG'] = _FakeConfig()
    app.config['VECTOR_STORE_SERVICE'] = _FakeVectorStoreService()

    data = {'file': (io.BytesIO(b'https://x.com/user/status/456\n'), 'urls.txt')}
    with app.test_client() as client:
        response = client.post('/prompt/demo/uploadx/batch', data=data, content_type='multipart/form-data')

    assert response.status_code == 500
    assert b'error processing x post' in response.data.lower() or b'boom' in response.data.lower()
