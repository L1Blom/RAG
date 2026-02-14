"""Configuration models for RAG service."""

from dataclasses import dataclass, field
from typing import Optional
import logging
from rag.config.settings import ConfigService
from rag.config.constants import DEFAULT_TEMPERATURE_MIN, DEFAULT_TEMPERATURE_MAX


@dataclass
class RAGConfig:
    """Configuration for RAG service instance."""
    
    # Project identification
    project: str
    
    # LLM settings
    use_llm: str
    model_text: str
    embedding_model: str
    temperature: float
    max_tokens: float
    
    # Retrieval settings
    similar: int
    score: float
    
    # Document processing
    chunk_size: int
    chunk_overlap: int
    
    # Prompts
    contextualize_q_system_prompt: str
    system_prompt: str
    
    # Directories
    data_dir: str
    persistence_dir: str
    html_dir: str
    
    # Flask settings
    port: int
    max_mb_size: int
    
    # Logging
    logging_level: str
    
    # Runtime state (not from config)
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_temperature()
        self._validate_paths()
    
    def _validate_temperature(self):
        """Ensure temperature is within valid range."""
        if not (DEFAULT_TEMPERATURE_MIN <= self.temperature <= DEFAULT_TEMPERATURE_MAX):
            raise ValueError(
                f"Temperature must be between {DEFAULT_TEMPERATURE_MIN} and "
                f"{DEFAULT_TEMPERATURE_MAX}, got {self.temperature}"
            )
    
    def _validate_paths(self):
        """Validate that required paths are set."""
        if not self.data_dir:
            raise ValueError("data_dir must be specified")
        if not self.persistence_dir:
            raise ValueError("persistence_dir must be specified")
    
    @classmethod
    def from_config_file(cls, config_service: ConfigService, project: str) -> 'RAGConfig':
        """
        Create RAGConfig from a ConfigService instance.
        
        Args:
            config_service: ConfigService instance with loaded configuration
            project: Project identifier
            
        Returns:
            RAGConfig instance
            
        Example:
            >>> config_service = ConfigService('constants/constants_myproject.ini')
            >>> rag_config = RAGConfig.from_config_file(config_service, 'myproject')
        """
        # Get LLM provider
        use_llm = config_service.get_string('LLMS', 'use_llm')
        llm_section = f'LLMS.{use_llm}'
        
        return cls(
            project=project,
            use_llm=use_llm,
            model_text=config_service.get_string(llm_section, 'modeltext'),
            embedding_model=config_service.get_string(llm_section, 'embedding_model'),
            temperature=config_service.get_float('DEFAULT', 'temperature'),
            max_tokens=config_service.get_float('DEFAULT', 'max_tokens'),
            similar=config_service.get_int('DEFAULT', 'similar'),
            score=config_service.get_float('DEFAULT', 'score'),
            chunk_size=config_service.get_int('DEFAULT', 'chunk_size'),
            chunk_overlap=config_service.get_int('DEFAULT', 'chunk_overlap'),
            contextualize_q_system_prompt=config_service.get_string(
                'DEFAULT', 'contextualize_q_system_prompt'
            ),
            system_prompt=config_service.get_string('DEFAULT', 'system_prompt'),
            data_dir=config_service.get_string('DEFAULT', 'data_dir'),
            persistence_dir=config_service.get_string('DEFAULT', 'persistence'),
            html_dir=config_service.get_string('DEFAULT', 'html'),
            port=config_service.get_int('FLASK', 'port'),
            max_mb_size=config_service.get_int('FLASK', 'max_mb_size'),
            logging_level=config_service.get_string('DEFAULT', 'logging_level')
        )
