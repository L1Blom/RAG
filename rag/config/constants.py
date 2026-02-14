"""Application-wide constants."""

# File upload settings
DEFAULT_MAX_MB_SIZE = 16
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'pptx', 'xlsx', 'html'}
MAX_FILENAME_LENGTH = 255

# Temperature bounds
DEFAULT_TEMPERATURE_MIN = 0.0
DEFAULT_TEMPERATURE_MAX = 2.0

# Service settings
DEFAULT_CONFIG_PORT = 8000
DEFAULT_RAG_PORT = 5000

# Secrets and environment
SECRETS_DIR = "/run/secrets"
ENV_FILE = "env/config.env"

# Logging levels
LOG_LEVELS = {
    'DEBUG': 'DEBUG',
    'INFO': 'INFO',
    'WARNING': 'WARNING',
    'ERROR': 'ERROR',
    'CRITICAL': 'CRITICAL'
}
