# RAG Service Refactoring Guide

This document provides comprehensive refactoring advice for the RAG Service codebase to improve maintainability, testability, and scalability.

## **Critical Issues**

### 1. **Monolithic [`ragservice.py`](ragservice.py:1) (42,241 chars)**

This file is extremely large and violates the Single Responsibility Principle. It handles:
- Flask routing
- LLM configuration
- Vector store management
- Document loading
- Chat history
- Multiple provider integrations

**Recommendation:** Split into modules:

```
rag/
├── __init__.py
├── app.py              # Flask app initialization
├── routes.py           # API endpoints
├── models/
│   ├── llm_factory.py  # LLM provider factory pattern
│   ├── embeddings.py   # Embedding functions
│   └── chat_models.py  # Chat model configurations
├── services/
│   ├── vector_store.py # Vector store operations
│   ├── document_loader.py # Document loading logic
│   └── chat_history.py # Chat history management
└── config/
    └── settings.py     # Configuration management
```

### 2. **Global State Management (Lines 197-221)**

Using [`globvars`](ragservice.py:197) dictionary for global state is error-prone and makes testing difficult.

**Recommendation:** Use a proper configuration class:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    project: str
    use_llm: str
    model_text: str
    embedding: str
    temperature: float
    similar: int
    score: float
    max_tokens: float
    chunk_size: int
    chunk_overlap: int
    
    @classmethod
    def from_config_parser(cls, rc: configparser.ConfigParser, project: str):
        # Load from config
        pass
```

### 3. **Hardcoded Match Statements (Lines 110-147, 249-292)**

The [`get_modelnames()`](ragservice.py:110) and [`set_chat_model()`](ragservice.py:249) functions use match statements that will grow with each new provider.

**Recommendation:** Use Strategy Pattern:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def get_model_names(self) -> tuple[list, list]:
        pass
    
    @abstractmethod
    def create_chat_model(self, config: RAGConfig):
        pass
    
    @abstractmethod
    def create_embeddings(self, config: RAGConfig):
        pass

class OpenAIProvider(LLMProvider):
    def get_model_names(self):
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_APIKEY'))
        models = client.models.list().data
        # ... implementation
        
class ProviderFactory:
    _providers = {
        'OPENAI': OpenAIProvider,
        'AZURE': AzureProvider,
        'GROQ': GroqProvider,
        'OLLAMA': OllamaProvider,
        'NEBUL': NebulProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        return cls._providers[provider_name]()
```

### 4. **Secret Management (Lines 56-72)**

Mixing environment variables and file-based secrets is confusing.

**Recommendation:** Use a unified secrets manager:

```python
from typing import Optional
import os

class SecretsManager:
    def __init__(self):
        self.secrets_dir = "/run/secrets"
        self.env_file = "env/config.env"
        
    def get_secret(self, key: str) -> Optional[str]:
        # Try Docker secrets first
        secret_path = os.path.join(self.secrets_dir, key.lower())
        if os.path.exists(secret_path):
            with open(secret_path, 'r') as f:
                return f.read().strip()
        
        # Fall back to environment variable
        return os.environ.get(key)
```

### 5. **Error Handling in [`configservice.py`](configservice.py:1)**

Multiple functions lack proper error handling and validation.

**Recommendation:** Add proper exception handling:

```python
class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

@app.route('/set', methods=['POST'])
@cross_origin()
def set():
    try:
        my_input = request.json
        if not my_input:
            raise ConfigError("Request body is empty")
            
        required_keys = ['project', 'port', 'description', 'provider', 'llm']
        missing_keys = [k for k in required_keys if k not in my_input]
        if missing_keys:
            raise ConfigError(f"Missing required keys: {missing_keys}")
        
        # ... rest of implementation
        
    except ConfigError as e:
        logging.error("Configuration error: %s", e)
        return make_response({'error': str(e)}, 400)
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        return make_response({'error': 'Internal server error'}, 500)
```

### 6. **Document Loading Logic (Lines 370-450+)**

The [`load_files()`](ragservice.py:370) function is too complex with nested match statements.

**Recommendation:** Use a document loader registry:

```python
from typing import Protocol

class DocumentLoaderStrategy(Protocol):
    def load(self, config: dict, vectorstore) -> int:
        """Load documents and return number of splits"""
        pass

class PDFLoaderStrategy:
    def load(self, config: dict, vectorstore) -> int:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        loader = PyPDFDirectoryLoader(
            path=config['data_dir'],
            glob=config['glob']
        )
        splits = loader.load_and_split()
        if splits:
            vectorstore.add_documents(splits)
        return len(splits)

class TextLoaderStrategy:
    def load(self, config: dict, vectorstore) -> int:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(
            path=config['data_dir'],
            glob=config['glob'],
            loader_cls=TextLoader,
            silent_errors=True,
            loader_kwargs=text_loader_kwargs
        )
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        if splits:
            vectorstore.add_documents(splits)
        return len(splits)

class DocumentLoaderRegistry:
    def __init__(self):
        self._loaders = {
            'pdf': PDFLoaderStrategy(),
            'txt': TextLoaderStrategy(),
            'html': HTMLLoaderStrategy(),
            'docx': WordLoaderStrategy(),
            'pptx': PowerPointLoaderStrategy(),
            'xlsx': ExcelLoaderStrategy(),
        }
    
    def load_documents(self, file_type: str, config: dict, vectorstore):
        if file_type == 'all':
            return sum(
                loader.load(config, vectorstore) 
                for loader in self._loaders.values()
            )
        return self._loaders[file_type].load(config, vectorstore)
```

## **Code Quality Issues**

### 7. **Magic Numbers and Strings**

Many hardcoded values throughout the code.

**Recommendation:** Use constants:

```python
# constants.py
DEFAULT_MAX_MB_SIZE = 16
DEFAULT_TEMPERATURE_MIN = 0.0
DEFAULT_TEMPERATURE_MAX = 2.0
DEFAULT_CONFIG_PORT = 8000
SECRETS_DIR = "/run/secrets"
ENV_FILE = "env/config.env"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'xlsx', 'html'}
MAX_FILENAME_LENGTH = 255
```

### 8. **Inconsistent Logging**

Mix of `print()` and `logging` statements.

**Recommendation:** Use logging consistently:

```python
# Replace all print() with logging
# Line 51: print("Error: argument missing -> ID")
logging.error("Error: argument missing -> ID")

# Line 89: print("No constants file found: " + constantsfile)
logging.error("No constants file found: %s", constantsfile)

# Line 274: print(f"Model: {rcmodel}...")
logging.info("Model: %s, endpoint: %s, api_version: %s", rcmodel, endpoint, api_version)
```

### 9. **Type Hints Missing**

Most functions lack type hints, making the code harder to understand and maintain.

**Recommendation:** Add comprehensive type hints:

```python
from typing import List, Dict, Optional, Tuple
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

def embedding_function() -> Embeddings:
    """Return an Embedding instance based on configured LLM provider."""
    # implementation

def load_files(vectorstore: Chroma, file_type: str) -> None:
    """Load files into the vector store."""
    # implementation

def get_modelnames(
    mode: str, 
    modeltext: str, 
    embedding_model: Optional[str] = None
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load all API keys and return possible model names."""
    # implementation

def read_secret(secret_path: str) -> str:
    """Read a secret from the secrets directory."""
    with open(secret_path, 'r') as file:
        return file.read().strip()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Memory history store."""
    # implementation
```

### 10. **Configuration File Parsing**

Direct [`configparser`](ragservice.py:82) usage scattered throughout makes testing difficult.

**Recommendation:** Create a configuration service:

```python
class ConfigService:
    def __init__(self, config_file: str):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_file)
        
    def get_string(self, section: str, key: str, default: str = None) -> str:
        try:
            return self.parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is not None:
                return default
            raise
    
    def get_int(self, section: str, key: str, default: int = None) -> int:
        try:
            return self.parser.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is not None:
                return default
            raise
        
    def get_float(self, section: str, key: str, default: float = None) -> float:
        try:
            return self.parser.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if default is not None:
                return default
            raise
    
    def has_option(self, section: str, key: str) -> bool:
        return self.parser.has_option(section, key)
```

## **Architecture Improvements**

### 11. **Dependency Injection**

Hard dependencies make testing difficult.

**Recommendation:** Use dependency injection:

```python
class RAGService:
    def __init__(
        self,
        config: RAGConfig,
        llm_provider: LLMProvider,
        vector_store: Chroma,
        secrets_manager: SecretsManager
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.secrets_manager = secrets_manager
    
    def process_prompt(self, prompt: str, session_id: str) -> str:
        """Process a user prompt and return the response."""
        # Implementation using injected dependencies
        pass
```

### 12. **Testing Structure**

Current tests are integration tests. Need unit tests.

**Recommendation:** Add unit tests:

```
tests/
├── unit/
│   ├── test_llm_providers.py
│   ├── test_document_loaders.py
│   ├── test_config_service.py
│   ├── test_secrets_manager.py
│   └── test_rag_service.py
├── integration/
│   ├── test_ragservice_api.py
│   └── test_configservice_api.py
└── conftest.py  # pytest fixtures
```

Example unit test:

```python
# tests/unit/test_config_service.py
import pytest
from rag.config.settings import ConfigService

def test_config_service_get_string():
    config = ConfigService('constants/constants__unittest.ini')
    assert config.get_string('DEFAULT', 'description') == 'Unit test configuration'

def test_config_service_get_int():
    config = ConfigService('constants/constants__unittest.ini')
    assert config.get_int('FLASK', 'port') == 8888

def test_config_service_missing_key_with_default():
    config = ConfigService('constants/constants__unittest.ini')
    assert config.get_string('NONEXISTENT', 'key', default='default_value') == 'default_value'
```

### 13. **API Response Standardization**

Inconsistent response formats across endpoints.

**Recommendation:** Use standard response models:

```python
from pydantic import BaseModel
from typing import Optional, Any
import time

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        super().__init__(**data)

@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    response = APIResponse(
        success=True,
        data={
            'timestamp': globvars['timestamp'], 
            'llm': globvars['ModelText']
        }
    )
    return response.dict()

@app.route('/prompt/<project>', methods=['GET', 'POST'])
@cross_origin()
def prompt(project):
    try:
        prompt_text = request.args.get('prompt') or request.form.get('prompt')
        if not prompt_text:
            return APIResponse(
                success=False,
                error="Prompt parameter is required"
            ).dict(), 400
        
        result = process_prompt(prompt_text)
        return APIResponse(
            success=True,
            data={'answer': result}
        ).dict()
    except Exception as e:
        logging.error("Error processing prompt: %s", e)
        return APIResponse(
            success=False,
            error=str(e)
        ).dict(), 500
```

## **Security Concerns**

### 14. **File Upload Security (Line 92-95)**

File upload size limit is good, but need more validation.

**Recommendation:** Add comprehensive file validation:

```python
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'xlsx', 'html'}
MAX_FILENAME_LENGTH = 255

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and
        len(filename) <= MAX_FILENAME_LENGTH
    )

def secure_upload(file) -> str:
    """Securely handle file upload."""
    if not file or not allowed_file(file.filename):
        raise ValueError("Invalid file type or filename")
    
    filename = secure_filename(file.filename)
    
    # Additional sanitization - remove any remaining dangerous characters
    filename = filename.replace('..', '')
    
    # Ensure unique filename to prevent overwrites
    base, ext = os.path.splitext(filename)
    counter = 1
    final_path = os.path.join(upload_dir, filename)
    while os.path.exists(final_path):
        filename = f"{base}_{counter}{ext}"
        final_path = os.path.join(upload_dir, filename)
        counter += 1
    
    return filename
```

### 15. **Path Traversal Protection**

Line 396 has basic protection but could be improved.

**Recommendation:** Use pathlib for safer path handling:

```python
from pathlib import Path

def validate_path(base_dir: Path, user_path: str) -> Path:
    """Ensure user_path is within base_dir to prevent path traversal attacks."""
    base_dir = Path(base_dir).resolve()
    full_path = (base_dir / user_path).resolve()
    
    # Check if the resolved path is within the base directory
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"Path traversal detected: {user_path}")
    
    return full_path

# Usage example
def serve_file(filename: str):
    try:
        safe_path = validate_path(Path(base_dir) / 'data', filename)
        return send_file(safe_path)
    except ValueError as e:
        logging.error("Security violation: %s", e)
        return make_response({'error': 'Invalid file path'}, 403)
```

### 16. **API Key Exposure**

Ensure API keys are never logged or returned in responses.

**Recommendation:** Add API key sanitization:

```python
def sanitize_for_logging(data: dict) -> dict:
    """Remove sensitive information from data before logging."""
    sensitive_keys = ['api_key', 'apikey', 'password', 'secret', 'token']
    sanitized = data.copy()
    
    for key in sanitized:
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = '***REDACTED***'
    
    return sanitized

# Usage
logging.info("Configuration: %s", sanitize_for_logging(config_dict))
```

## **Performance Optimizations**

### 17. **Lazy Loading**

Load models and embeddings only when needed.

**Recommendation:** Implement lazy initialization:

```python
class LazyLLM:
    def __init__(self, provider: LLMProvider, config: RAGConfig):
        self._provider = provider
        self._config = config
        self._llm = None
    
    @property
    def llm(self):
        if self._llm is None:
            logging.info("Initializing LLM model: %s", self._config.model_text)
            self._llm = self._provider.create_chat_model(self._config)
        return self._llm
    
    def reset(self):
        """Reset the LLM instance (useful for model changes)."""
        self._llm = None
```

### 18. **Caching**

Add caching for frequently accessed data.

**Recommendation:** Use functools.lru_cache:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def get_embedding_for_text(text: str) -> tuple:
    """Cache embeddings for frequently used text."""
    embedding = embedding_function().embed_query(text)
    return tuple(embedding)  # Convert to tuple for hashability

@lru_cache(maxsize=32)
def load_config_file(config_path: str) -> dict:
    """Cache configuration file contents."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return dict(config)
```

### 19. **Database Connection Pooling**

For vector store operations, consider connection pooling.

**Recommendation:** Implement connection pooling:

```python
from contextlib import contextmanager

class VectorStorePool:
    def __init__(self, max_connections: int = 5):
        self._pool = []
        self._max_connections = max_connections
        self._lock = threading.Lock()
    
    @contextmanager
    def get_connection(self, config: RAGConfig):
        """Get a vector store connection from the pool."""
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = self._create_connection(config)
        
        try:
            yield conn
        finally:
            with self._lock:
                if len(self._pool) < self._max_connections:
                    self._pool.append(conn)
    
    def _create_connection(self, config: RAGConfig):
        return Chroma(
            persist_directory=config.persistence_dir,
            embedding_function=embedding_function()
        )
```

## **Documentation Improvements**

### 20. **Add Docstrings**

Many functions lack proper documentation.

**Recommendation:** Add comprehensive docstrings:

```python
def get_modelnames(
    mode: str, 
    modeltext: str, 
    embedding_model: Optional[str] = None
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load all API keys to environment variables and return possible model names.
    
    Args:
        mode: The LLM provider mode (e.g., 'OPENAI', 'AZURE', 'GROQ')
        modeltext: The specific model name to validate
        embedding_model: Optional embedding model name to validate
    
    Returns:
        A tuple containing:
        - List of available model names
        - List of available embedding names
        - List of Azure AI model names (empty for non-Azure providers)
        - List of Azure OpenAI model names (empty for non-Azure providers)
    
    Raises:
        SystemExit: If the specified model or embedding is not found
    
    Example:
        >>> models, embeddings, ai_models, openai_models = get_modelnames(
        ...     'OPENAI', 'gpt-4o', 'text-embedding-3-small'
        ... )
    """
    # Implementation
```

### 21. **API Documentation**

Add OpenAPI/Swagger documentation.

**Recommendation:** Use Flask-RESTX or similar:

```python
from flask_restx import Api, Resource, fields

api = Api(
    app,
    version='1.0',
    title='RAG Service API',
    description='A Retrieval-Augmented Generation service for document Q&A'
)

ns = api.namespace('prompt', description='Prompt operations')

prompt_model = api.model('Prompt', {
    'prompt': fields.String(required=True, description='The user prompt'),
    'session_id': fields.String(description='Session ID for conversation history')
})

@ns.route('/<string:project>')
class PromptResource(Resource):
    @api.doc('process_prompt')
    @api.expect(prompt_model)
    def post(self, project):
        """Process a prompt and return the AI response"""
        # Implementation
```

## **Priority Refactoring Order**

### Phase 1: High Priority (Foundation)
1. **Split [`ragservice.py`](ragservice.py:1) into modules** - This is the most critical change
2. **Replace global state with configuration classes** - Essential for testability
3. **Implement Strategy Pattern for LLM providers** - Improves extensibility
4. **Add comprehensive error handling** - Critical for production stability
5. **Standardize logging** - Replace all `print()` statements

### Phase 2: Medium Priority (Quality)
1. **Add type hints throughout** - Improves code clarity and IDE support
2. **Implement dependency injection** - Makes testing much easier
3. **Add unit tests** - Essential for maintaining code quality
4. **Create configuration service** - Centralizes config management
5. **Standardize API responses** - Improves API consistency

### Phase 3: Low Priority (Enhancement)
1. **Performance optimizations** - Lazy loading, caching
2. **Enhanced security measures** - Additional validation and sanitization
3. **Add comprehensive documentation** - Docstrings and API docs
4. **Implement connection pooling** - For better resource management
5. **Add monitoring and metrics** - For production observability

## **Migration Strategy**

To avoid breaking existing functionality, follow this migration approach:

1. **Create new module structure alongside existing code**
2. **Gradually migrate functionality to new modules**
3. **Keep old code working until migration is complete**
4. **Add tests for each migrated component**
5. **Remove old code only after thorough testing**

Example migration for LLM providers:

```python
# Step 1: Create new provider system
# rag/models/llm_factory.py
class ProviderFactory:
    # New implementation

# Step 2: Add compatibility layer in ragservice.py
def set_chat_model_legacy(temp=rctemp):
    """Legacy function - delegates to new provider system"""
    provider = ProviderFactory.get_provider(globvars['USE_LLM'])
    globvars['LLM'] = provider.create_chat_model(config)

# Step 3: Update callers gradually
# Old: set_chat_model(0.7)
# New: provider.create_chat_model(config)

# Step 4: Remove legacy function after all callers updated
```

## **Conclusion**

This refactoring will transform the codebase into a more maintainable, testable, and scalable system. The key benefits include:

- **Better separation of concerns** - Each module has a single responsibility
- **Improved testability** - Dependency injection and modular design enable unit testing
- **Enhanced extensibility** - Strategy pattern makes adding new providers easy
- **Increased reliability** - Better error handling and validation
- **Better developer experience** - Type hints, documentation, and clear structure

Start with Phase 1 high-priority items and gradually work through the remaining improvements. Each change should be accompanied by tests to ensure no regression in functionality.
