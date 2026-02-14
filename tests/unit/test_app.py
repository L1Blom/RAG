"""Unit tests for the Flask app factory and middleware."""

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
