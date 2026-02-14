"""Main Flask application for RAG service.

This module provides the application factory ``create_app`` and the CLI
entry point ``main``.  It wires together configuration, secrets, services
and the Flask blueprint created in ``rag.api.routes``.
"""

import os
import sys
import uuid
import time
import json
import logging
from urllib.request import urlopen

# Silence LangChain "USER_AGENT not set" warning
os.environ.setdefault("USER_AGENT", "RAG-service/1.0")

from flask import Flask

from rag.config.settings import ConfigService
from rag.config.secrets import SecretsManager
from rag.models.config_models import RAGConfig
from rag.models.llm.factory import ProviderFactory
from rag.services.embeddings import EmbeddingsService
from rag.services.chat_history import ChatHistoryService
from rag.services.vector_store import VectorStoreService
from rag.api.routes import rag_bp, initialize_chain
from rag.api.middleware import setup_middleware
from rag.utils.logging_config import setup_logging


def _set_chat_model(config: RAGConfig, config_service: ConfigService, state: dict):
    """
    Create a closure that (re-)creates the chat model and stores it in *state*.

    The returned callable accepts an optional *temperature* override and is
    stored in ``state['set_chat_model']`` so that route handlers can call it
    when the user changes the model or temperature at runtime.
    """
    def _inner(temperature=None):
        temp = temperature if temperature is not None else config.temperature
        provider = ProviderFactory.get_provider(
            config.use_llm, config, config_service=config_service
        )
        state['LLM'] = provider.create_chat_model(temp)

    return _inner


def create_app(project: str) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        project: Project identifier (maps to ``constants/constants_{project}.ini``)

    Returns:
        Configured Flask application ready to serve.
    """
    # ── Secrets ──────────────────────────────────────────────────────────
    secrets_manager = SecretsManager()
    secrets_manager.load_all_secrets()

    # ── Configuration ────────────────────────────────────────────────────
    config_file = f"constants/constants_{project}.ini"
    config_service = ConfigService(config_file)
    rag_config = RAGConfig.from_config_file(config_service, project)

    # ── Logging ──────────────────────────────────────────────────────────
    setup_logging(rag_config.logging_level)
    logging.info("Starting RAG service for project: %s", project)
    logging.info("Working directory is %s", os.getcwd())

    # Silence werkzeug unless there is an error
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # ── Flask ────────────────────────────────────────────────────────────
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = rag_config.max_mb_size * 1024 * 1024

    # ── Middleware ────────────────────────────────────────────────────────
    setup_middleware(app)

    # ── Services ─────────────────────────────────────────────────────────
    embeddings_service = EmbeddingsService(rag_config, config_service=config_service)
    chat_history_service = ChatHistoryService()
    vector_store_service = VectorStoreService(rag_config, embeddings_service)

    # ── Model discovery ──────────────────────────────────────────────────
    provider = ProviderFactory.get_provider(
        rag_config.use_llm, rag_config, config_service=config_service
    )
    modelnames, embeddingnames = provider.get_model_names()

    # Validate chosen model / embeddings
    if rag_config.model_text not in modelnames:
        logging.error("Model %s not found in %s models", rag_config.model_text, rag_config.use_llm)
        sys.exit(os.EX_CONFIG)
    if rag_config.embedding_model not in embeddingnames:
        logging.error("Embedding %s not found in %s models", rag_config.embedding_model, rag_config.use_llm)
        sys.exit(os.EX_CONFIG)

    # ── Mutable runtime state (equivalent to old globvars) ───────────────
    state = {
        'Project': project,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'USE_LLM': rag_config.use_llm,
        'ModelText': rag_config.model_text,
        'Embedding': rag_config.embedding_model,
        'Temperature': rag_config.temperature,
        'Similar': rag_config.similar,
        'Score': rag_config.score,
        'Tokens': rag_config.max_tokens,
        'ChunkSize': rag_config.chunk_size,
        'ChunkOverlap': rag_config.chunk_overlap,
        'Prompt': rag_config.system_prompt,
        'Session': uuid.uuid4(),
        'Chain': None,
        'LLM': None,
        'NoChunks': 0,
        'modelnames': modelnames,
        'embeddingnames': embeddingnames,
    }

    # Create the set_chat_model helper and call it once
    set_chat = _set_chat_model(rag_config, config_service, state)
    state['set_chat_model'] = set_chat
    set_chat(rag_config.temperature)

    # ── Store everything in app.config so routes can access it ───────────
    app.config['RAG_CONFIG'] = rag_config
    app.config['CONFIG_SERVICE'] = config_service
    app.config['EMBEDDINGS_SERVICE'] = embeddings_service
    app.config['CHAT_HISTORY_SERVICE'] = chat_history_service
    app.config['VECTOR_STORE_SERVICE'] = vector_store_service
    app.config['APP_STATE'] = state

    # ── Register blueprint ───────────────────────────────────────────────
    app.register_blueprint(rag_bp)

    # ── Initialize the RAG chain ─────────────────────────────────────────
    with app.app_context():
        initialize_chain()

    logging.info("RAG service initialized successfully")
    return app


def main():
    """CLI entry point – expects one argument: the project ID."""
    if len(sys.argv) != 2:
        logging.error("Error: argument missing -> ID")
        sys.exit(os.EX_USAGE)

    project = sys.argv[1]
    app = create_app(project)

    # Optionally try to get port from a config server
    config_service = app.config['CONFIG_SERVICE']
    rag_config = app.config['RAG_CONFIG']
    port = rag_config.port
    debug = False

    try:
        config_host = config_service.get_string('DEFAULT', 'config_server', default='')
        if config_host:
            response = urlopen(config_host + "/get?project=" + project)
            configs = json.loads(response.read().decode('utf-8'))
            port = int(configs['port'])
            logging.info("Port found at config server: %s", port)
    except Exception as e:
        logging.error("Failed to get port from config server: %s", e)
        logging.info("Using port from constants file: %s", port)

    try:
        debug = config_service.get_boolean('FLASK', 'debug', default=False)
    except Exception:
        pass

    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
