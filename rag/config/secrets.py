"""Unified secrets management for API keys and credentials."""

import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from .constants import SECRETS_DIR, ENV_FILE


class SecretsManager:
    """Manages secrets from Docker secrets and environment variables."""
    
    def __init__(self, secrets_dir: str = SECRETS_DIR, env_file: str = ENV_FILE):
        """
        Initialize the secrets manager.
        
        Args:
            secrets_dir: Directory containing Docker secrets
            env_file: Path to .env file
        """
        self.secrets_dir = Path(secrets_dir)
        self.env_file = env_file
        self._load_environment()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file if it exists."""
        if os.path.exists(self.env_file):
            logging.info("Loading environment variables from %s", self.env_file)
            load_dotenv(self.env_file)
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get a secret value, checking Docker secrets first, then environment variables.
        
        Args:
            key: The secret key to retrieve
            
        Returns:
            The secret value, or None if not found
            
        Example:
            >>> manager = SecretsManager()
            >>> api_key = manager.get_secret('OPENAI_APIKEY')
        """
        # Try Docker secrets first
        secret_path = self.secrets_dir / key.lower()
        if secret_path.exists():
            try:
                return secret_path.read_text().strip()
            except Exception as e:
                logging.error("Error reading secret from %s: %s", secret_path, e)
        
        # Fall back to environment variable
        return os.environ.get(key)
    
    def load_all_secrets(self) -> None:
        """Load all known secrets into environment variables."""
        secret_keys = [
            'AZURE_AI_APIKEY',
            'AZURE_OPENAI_APIKEY',
            'OPENAI_APIKEY',
            'OLLAMA_APIKEY',
            'GROQ_APIKEY',
            'NEBUL_APIKEY'
        ]
        
        for key in secret_keys:
            value = self.get_secret(key)
            if value:
                os.environ[key] = value
                logging.debug("Loaded secret: %s", key)
