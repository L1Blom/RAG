"""Flask API routes for RAG service.

Extracts all routes from the original ragservice.py into a clean Blueprint.
Services are accessed via Flask's current_app.config.
"""

import os
import re
import json
import uuid
import base64
import logging
import pprint
from pathlib import PurePath
from urllib.request import urlopen
from urllib.parse import quote

from flask import Blueprint, request, make_response, send_file, current_app, jsonify
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from rag.models.response_models import success_response, error_response
from rag.utils.exceptions import RAGException, ValidationError
from rag.utils.security import allowed_file, validate_path


# Create blueprint
rag_bp = Blueprint('rag', __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_service(name: str):
    """Get a service from the app config."""
    return current_app.config[name]


def _get_config():
    """Get the RAGConfig instance."""
    return current_app.config['RAG_CONFIG']


def _get_state():
    """Get the mutable runtime state dict."""
    return current_app.config['APP_STATE']


def _log_error(error_text: str):
    """Log error and raise HTTPException."""
    logging.error(error_text)
    raise HTTPException(error_text)


def _encode_image(image_url: str) -> str:
    """Encode an image from URL to base64."""
    logging.info("Encoding image: %s", image_url)
    with urlopen(image_url) as url:
        f = url.read()
        image = base64.b64encode(f).decode("utf-8")
    return image


# ---------------------------------------------------------------------------
# Chain management
# ---------------------------------------------------------------------------

def initialize_chain(new_vectorstore: bool = False) -> bool:
    """Initialize the RAG chain using injected services."""
    config = _get_config()
    state = _get_state()
    vector_store_svc = _get_service('VECTOR_STORE_SERVICE')
    chat_history_svc = _get_service('CHAT_HISTORY_SERVICE')
    config_service = _get_service('CONFIG_SERVICE')

    chat_model = state['LLM']

    if new_vectorstore:
        vector_store_svc.reload_documents()

    vectorstore = vector_store_svc.vectorstore
    state['NoChunks'] = vectorstore._collection.count()

    search_type = config_service.get_string('DEFAULT', 'search_type', default='similarity_score_threshold')
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            'score_threshold': config.score,
            'k': 8
        }
    )

    # Contextualize question
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", config.contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # Answer question
    system_prompt = state['Prompt'] + "{context}"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    state['Chain'] = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=chat_history_svc.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    logging.info("Chain initialized: %s", config.model_text)
    return True


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@rag_bp.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    """Health check endpoint."""
    state = _get_state()
    return {'answer': {
        'timestamp': state.get('timestamp', ''),
        'llm': _get_config().model_text,
    }}


# ---------------------------------------------------------------------------
# Core prompt routes
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>', methods=['GET', 'POST'])
@cross_origin()
def prompt(project):
    """Process a prompt and return the answer text."""
    state = _get_state()
    try:
        if request.method == 'GET':
            prompt_text = request.values.get('prompt')
        else:
            prompt_text = request.form.get('prompt')
        if not prompt_text:
            return make_response("Prompt parameter is required", 400)

        answer = state['Chain'].invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": str(state['Session'])}},
        )
        logging.info("Result prompt: %s", answer['answer'])
        return make_response(answer['answer'], 200)
    except HTTPException as e:
        logging.error("Error processing prompt: %s", e)
        return make_response(f"Error processing prompt, due to {e}", 500)
    except Exception as e:
        logging.error("Unexpected error in prompt: %s", e)
        return make_response(f"Error processing prompt, due to {e}", 500)


@rag_bp.route('/prompt/<project>/full', methods=['GET', 'POST'])
@cross_origin()
def prompt_full(project):
    """Process a prompt and return answer with context."""
    state = _get_state()
    try:
        if request.method == 'GET':
            prompt_text = request.values.get('prompt')
        else:
            prompt_text = request.form.get('prompt')
        if not prompt_text:
            return make_response("Prompt parameter is required", 400)

        result = state['Chain'].invoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": str(state['Session'])}},
        )
        mylist = [item.to_json() for item in result['context']]
        myanswer = result['answer']
        mylist = pprint.pformat(
            object=[item['kwargs'] for item in mylist], width=132
        )
        return "\n".join([mylist, myanswer, ""])
    except HTTPException as e:
        logging.error("Error processing full prompt: %s", e)
        return make_response(f"Error processing prompt, due to {e}", 500)


# ---------------------------------------------------------------------------
# Search & document routes
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/search', methods=['GET', 'POST'])
@cross_origin()
def search(project):
    """Search the vector store for similar documents."""
    state = _get_state()
    config = _get_config()
    try:
        if request.method == 'GET':
            prompt_text = request.values.get('prompt')
            sim = request.values.get('similar', config.similar)
        else:
            prompt_text = request.form.get('prompt')
            sim = request.form.get('similar', config.similar)
        sim = int(sim)

        vector_store_svc = _get_service('VECTOR_STORE_SERVICE')
        result = vector_store_svc.vectorstore.similarity_search_with_relevance_scores(
            prompt_text, sim
        )
        mylist = [
            {'page_content': item[0].page_content,
             'metadata': item[0].metadata,
             'score': item[1]}
            for item in result
        ]
        return mylist
    except Exception as e:
        logging.error("Error in search: %s", e)
        return make_response(f"Error in search: {e}", 500)


@rag_bp.route('/prompt/<project>/documents', methods=['GET'])
@cross_origin()
def documents(project):
    """Get documents from the vector store."""
    vector_store_svc = _get_service('VECTOR_STORE_SERVICE')
    doc_id = request.values.get('id', '')
    if doc_id:
        result = vector_store_svc.vectorstore.get(doc_id)
    else:
        result = vector_store_svc.vectorstore.get()
    return {str(k): str(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Configuration query routes
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/params', methods=['GET'])
@cross_origin()
def parameters(project):
    """Return a config parameter value."""
    section = request.values.get('section')
    param = request.values.get('param')
    config_service = _get_service('CONFIG_SERVICE')
    answer = config_service.get_string(section, param)
    return make_response(answer, 200)


@rag_bp.route('/prompt/<project>/globals', methods=['GET'])
@cross_origin()
def globals_route(project):
    """Return the runtime state dict."""
    state = _get_state()
    return {str(k): str(v) for k, v in state.items()}


@rag_bp.route('/prompt/<project>/modelnames', methods=['GET'])
@cross_origin()
def model_names(project):
    """Return available model names."""
    state = _get_state()
    return jsonify(state.get('modelnames', []))


@rag_bp.route('/prompt/<project>/embeddingnames', methods=['GET'])
@cross_origin()
def embedding_names(project):
    """Return available embedding names."""
    state = _get_state()
    return jsonify(state.get('embeddingnames', []))


# ---------------------------------------------------------------------------
# Runtime configuration mutation routes
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/model', methods=['GET', 'POST'])
@cross_origin()
def set_model(project):
    """Set the LLM model."""
    state = _get_state()
    config = _get_config()
    if request.method == 'GET':
        this_model = request.values.get('model')
    else:
        this_model = request.form.get('model')

    if not this_model or not this_model.isascii():
        _log_error(f"Model name {this_model} is invalid or uses non-ascii characters")

    if this_model not in state.get('modelnames', []):
        _log_error(f"Model {this_model} not found in available models")

    config.model_text = this_model
    state['set_chat_model'](config.temperature)
    initialize_chain(True)
    return make_response(f'Model set to {this_model}', 200)


@rag_bp.route('/prompt/<project>/embeddings', methods=['GET', 'POST'])
@cross_origin()
def set_embeddings(project):
    """Set the embedding model."""
    state = _get_state()
    config = _get_config()
    if request.method == 'GET':
        this_embedding = request.values.get('embedding')
    else:
        this_embedding = request.form.get('embedding')

    if not this_embedding or not this_embedding.isascii():
        _log_error(f"Embedding name {this_embedding} is invalid")

    if this_embedding not in state.get('embeddingnames', []):
        _log_error(f"Embedding {this_embedding} not found in available models")

    config.embedding_model = this_embedding
    state['set_chat_model'](config.temperature)
    initialize_chain(True)
    return make_response(f'Embedding set to {this_embedding}', 200)


@rag_bp.route('/prompt/<project>/chunk', methods=['GET', 'POST'])
@cross_origin()
def set_chunk(project):
    """Set chunk size and overlap."""
    config = _get_config()
    if request.method == 'GET':
        chunk_size = int(request.values.get('chunk_size'))
        chunk_overlap = int(request.values.get('chunk_overlap'))
    else:
        chunk_size = int(request.form.get('chunk_size'))
        chunk_overlap = int(request.form.get('chunk_overlap'))

    if chunk_size < 1 or chunk_size > 10000:
        _log_error(f"Chunk size {chunk_size} not between 1 and 10000")
    if chunk_overlap < 0 or chunk_overlap > 1000:
        _log_error(f"Chunk overlap {chunk_overlap} not between 0 and 1000")

    config.chunk_size = chunk_size
    config.chunk_overlap = chunk_overlap
    initialize_chain(True)
    return make_response(f'Chunk set to {chunk_size} with overlap {chunk_overlap}', 200)


@rag_bp.route('/prompt/<project>/temp', methods=['GET', 'POST'])
@cross_origin()
def set_temperature(project):
    """Set the temperature."""
    state = _get_state()
    config = _get_config()
    if request.method == 'GET':
        temperature = float(request.values.get('temp'))
    else:
        temperature = float(request.form.get('temp'))

    if temperature < 0.0 or temperature > 2.0:
        _log_error(f"Temperature {temperature} not between 0.0 and 2.0")

    config.temperature = temperature
    state['set_chat_model'](temperature)
    initialize_chain(True)
    return make_response(f'Temperature set to {temperature}', 200)


@rag_bp.route('/prompt/<project>/systemprompt', methods=['GET', 'POST'])
@cross_origin()
def set_system_prompt(project):
    """Set the system prompt."""
    state = _get_state()
    if request.method == 'GET':
        new_prompt = request.values.get('systemprompt')
    else:
        new_prompt = request.form.get('systemprompt')

    state['Prompt'] = new_prompt
    initialize_chain(True)
    return make_response(f'System prompt set to {new_prompt}', 200)


# ---------------------------------------------------------------------------
# Reload / clear / cache
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/reload', methods=['GET', 'POST'])
@cross_origin()
def reload(project):
    """Reload documents into the vector store."""
    initialize_chain(True)
    return make_response('Documents reloaded', 200)


@rag_bp.route('/prompt/<project>/clear', methods=['GET', 'POST'])
@cross_origin()
def clear(project):
    """Clear chat history."""
    chat_history_svc = _get_service('CHAT_HISTORY_SERVICE')
    chat_history_svc.clear_all()
    return make_response('History deleted', 200)


@rag_bp.route('/prompt/<project>/cache', methods=['GET', 'POST'])
@cross_origin()
def cache(project):
    """Return the chat history cache contents."""
    state = _get_state()
    chat_history_svc = _get_service('CHAT_HISTORY_SERVICE')
    content = ""
    session_id = str(state['Session'])
    if session_id in chat_history_svc.get_all_sessions():
        history = chat_history_svc.get_session_history(session_id)
        for message in history.messages:
            prefix = "AI" if isinstance(message, AIMessage) else "User"
            content += f"{prefix}:{message}\n"
    return make_response(content, 200)


# ---------------------------------------------------------------------------
# File serving & context management
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/file', methods=['GET'])
@cross_origin()
def send_files(project):
    """Serve a file from the data directory."""
    config = _get_config()
    filename = request.values.get('file')
    data_dir = config.data_dir
    base_dir = os.path.abspath(os.path.dirname(__file__) + '/../../') + '/'

    absolute_path = data_dir.startswith('/')
    if absolute_path:
        serve_file = os.path.normpath(os.path.join(data_dir, filename))
    else:
        serve_file = os.path.normpath(os.path.join(base_dir + data_dir, filename))

    if not serve_file.startswith(base_dir):
        logging.error("Parameter value for file path not allowed")

    logging.info("File served: %s", serve_file)
    if os.path.exists(serve_file):
        return send_file(serve_file)
    else:
        return make_response("", 200)


@rag_bp.route('/prompt/<project>/context', methods=['GET'])
@cross_origin()
def list_files(project):
    """List or delete context files/URLs."""
    config = _get_config()
    filename = request.values.get('file', '')
    action = request.values.get('action', 'list')
    mode = request.values.get('mode', 'file')
    base_dir = os.path.abspath(os.path.dirname(__file__) + '/../../') + '/'

    if action not in ['list', 'delete']:
        logging.error("Action %s not allowed, only list or delete", action)

    context_dir = config.data_dir
    serve_files = os.path.normpath(os.path.join(base_dir, context_dir))
    if not serve_files.startswith(base_dir):
        logging.error("Parameter value for DATA_DIR not allowed")

    if mode == 'file':
        all_files = os.listdir(serve_files)
        output = {"name": "Context files", "type": "folder", "items": []}
        if action == 'delete':
            os.remove(os.path.join(serve_files, filename))
        context_files = [
            f for f in all_files if os.path.isfile(os.path.join(serve_files, f))
        ]
        for f in context_files:
            output['items'].append({"name": f, "type": "file"})

    elif mode == 'url':
        urls = _load_urls(serve_files)
        if urls is None:
            urls = []
        output = {"name": "Context URLs", "type": "urls", "items": []}
        if action == 'delete' and filename in urls:
            urls.remove(filename)
            urls_file = os.path.join(serve_files, 'urls.json')
            with open(urls_file, 'w') as f:
                json.dump(urls, f, indent=4)
        for url in urls:
            output['items'].append({"name": url, "type": "url"})
    else:
        output = {"name": "Unknown mode", "type": "error", "items": []}

    return output


# ---------------------------------------------------------------------------
# File & URL upload
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/upload', methods=['POST'])
@cross_origin()
def upload_file(project):
    """Handle file upload."""
    config = _get_config()
    config_service = _get_service('CONFIG_SERVICE')
    state = _get_state()

    if 'file' not in request.files:
        _log_error("No file part in the request")

    file = request.files['file']
    if file.filename == '':
        _log_error("No filename found in upload")

    filename = secure_filename(file.filename)

    # Check extension against configured globs
    found = any(
        PurePath(filename).match(
            config_service.get_string('DEFAULT', glob_key, default='')
        )
        for glob_key in [
            'data_glob_txt', 'data_glob_pdf', 'data_glob_docx',
            'data_glob_xlsx', 'data_glob_pptx', 'data_glob_html'
        ]
    )

    if not found:
        _log_error("File extension doesn't match configured GLOB patterns")

    filepath = os.path.join(config.data_dir, filename)
    logging.info("Saving %s to: %s", filename, filepath)
    file.save(filepath)
    logging.info("File %s saved to: %s", filename, filepath)

    # Load the new file into the vector store
    vector_store_svc = _get_service('VECTOR_STORE_SERVICE')
    file_ext = filename.rsplit('.', 1)[-1].lower()
    vector_store_svc.reload_documents(file_ext)

    return make_response('File stored successfully', 200)


@rag_bp.route('/prompt/<project>/uploadurl', methods=['GET'])
@cross_origin()
def upload_url(project):
    """Add a URL to the context."""
    config = _get_config()
    base_dir = os.path.abspath(os.path.dirname(__file__) + '/../../') + '/'
    serve_files = os.path.normpath(os.path.join(base_dir, config.data_dir))

    url = request.values.get('data')
    urls = _load_urls(serve_files)
    if urls is None:
        urls = []

    url_pattern = re.compile(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', re.IGNORECASE)
    if url_pattern.match(url):
        urls.append(url)
        urls_file = os.path.join(serve_files, 'urls.json')
        with open(urls_file, 'w') as f:
            json.dump(urls, f, indent=4)

        vector_store_svc = _get_service('VECTOR_STORE_SERVICE')
        vector_store_svc.reload_documents('html')
        logging.info("URL %s stored successfully", url)

    return make_response('URL stored successfully', 200)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

@rag_bp.route('/prompt/<project>/image', methods=['GET', 'POST'])
@cross_origin()
def process_image(project):
    """Send an image to the LLM for analysis."""
    config = _get_config()
    state = _get_state()
    config_service = _get_service('CONFIG_SERVICE')

    if request.method == 'GET':
        my_url = request.values.get('image')
        text = request.values.get('prompt')
    else:
        my_url = request.form.get('image')
        text = request.form.get('prompt')

    image_url = quote(my_url, safe='/:?=&')
    logging.info("Processing image: %s, with prompt: %s", image_url, text)
    bimage = _encode_image(image_url)

    use_llm = config.use_llm
    if use_llm in ('OPENAI', 'NEBUL'):
        api_key_var = 'NEBUL_APIKEY' if use_llm == 'NEBUL' else 'OPENAI_APIKEY'
        kwargs = {
            'api_key': os.environ.get(api_key_var),
            'model': config.model_text,
            'temperature': config.temperature,
        }
        if use_llm == 'NEBUL':
            kwargs['base_url'] = config_service.get_string('LLMS.NEBUL', 'base_url')
        chain = ChatOpenAI(**kwargs)
        msg = chain.invoke([
            AIMessage(content="Picture revealer"),
            HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{bimage}"}}
            ])
        ])
        return make_response(msg.content, 200)

    elif use_llm == 'AZURE':
        from azure.ai.inference.models import (
            ImageContentItem, TextContentItem, ImageUrl,
            UserMessage, SystemMessage
        )
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        messages = [
            SystemMessage(content="Picture revealer"),
            UserMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{bimage}"}}
            ])
        ]
        chain = ChatCompletionsClient(
            credential=AzureKeyCredential(os.environ.get('AZURE_AI_APIKEY')),
            endpoint=config_service.get_string('LLMS.AZURE.AI', 'model_endpoint'),
            model=config.model_text
        )
        msg = chain.complete(messages=messages, response_format="json_object", stream=False)
        return make_response(msg.get("choices")[0].get("message").get("content"), 200)

    return make_response('Image processing not supported for this LLM provider', 200)


# ---------------------------------------------------------------------------
# URL helper
# ---------------------------------------------------------------------------

def _load_urls(directory: str):
    """Load URLs from urls.json in the given directory."""
    urls_file = os.path.join(directory, 'urls.json')
    if not os.path.exists(urls_file):
        return None
    try:
        with open(urls_file, 'r') as f:
            contents = f.read()
            if contents.strip() == "":
                return []
            contents = json.loads(contents)
            if not isinstance(contents, list):
                logging.error("URLs file does not contain a list")
                return None
            logging.info("Found %d URLs in file", len(contents))
            return contents
    except json.JSONDecodeError as e:
        logging.error("Error reading URLs file: %s", e)
        return None
