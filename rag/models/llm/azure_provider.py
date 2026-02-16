"""Azure LLM provider implementation.

Azure has two distinct sub-platforms, each with its own endpoint style:

* **Azure OpenAI** – model endpoint = ``<base>/openai/deployments/<model>``
  The INI stores the base (``model_endpoint``) and we append the model name.
  Embedding endpoint is stored fully qualified in ``embedding_endpoint``.
* **Azure AI** – model endpoint is already fully qualified in ``model_endpoint``.

The INI sections are ``LLMS.AZURE.OPENAI`` and ``LLMS.AZURE.AI``.
"""

import os
import logging
from typing import List, Tuple, Optional
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from .base import LLMProvider
from rag.config.settings import ConfigService


class AzureProvider(LLMProvider):
    """Azure provider implementation (supports both Azure OpenAI and Azure AI)."""
    
    def __init__(self, config, config_service: Optional[ConfigService] = None):
        super().__init__(config)
        self._config_service = config_service
        self._modelnames_ai = []
        self._modelnames_openai = []
    
    def _get_config_service(self) -> ConfigService:
        if self._config_service is None:
            raise ValueError("ConfigService required for Azure provider")
        return self._config_service
    
    def get_model_names(self) -> Tuple[List[str], List[str]]:
        """Get available Azure models and embeddings from config."""
        cs = self._get_config_service()
        self._modelnames_ai = cs.get_list('LLMS.AZURE.AI', 'models', default=[])
        self._modelnames_openai = cs.get_list('LLMS.AZURE.OPENAI', 'models', default=[])
        model_names = self._modelnames_ai + self._modelnames_openai
        embedding_names = cs.get_list('LLMS.AZURE.OPENAI', 'embeddings', default=[])
        return model_names, embedding_names
    
    def _get_azure_settings(self):
        """Get Azure endpoint and credential settings.

        Returns a tuple of:
            (api_key, model_api_version, model_endpoint,
             embedding_api_key, embedding_api_version, embedding_endpoint)
        """
        cs = self._get_config_service()
        model = self.config.model_text
        embedding = self.config.embedding_model

        # Ensure model name lists are populated
        if not self._modelnames_ai and not self._modelnames_openai:
            self.get_model_names()
        
        api_key = None
        model_api_version = None
        embedding_api_version = None
        embedding_api_key = None
        model_endpoint = None
        embedding_endpoint = None
        
        # --- Chat model settings ---
        if model in self._modelnames_ai:
            api_key = os.environ.get('AZURE_AI_APIKEY')
            model_endpoint = cs.get_string('LLMS.AZURE.AI', 'model_endpoint')
            model_api_version = cs.get_string('LLMS.AZURE.AI', 'model_api_version')
        
        if model in self._modelnames_openai:
            api_key = os.environ.get('AZURE_OPENAI_APIKEY')
            # Append model name to base endpoint, same as old code
            model_endpoint = cs.get_string('LLMS.AZURE.OPENAI', 'model_endpoint') + model
            model_api_version = cs.get_string('LLMS.AZURE.OPENAI', 'model_api_version')
        
        # --- Embedding settings (always Azure OpenAI) ---
        _, embedding_names = self.get_model_names()
        if embedding in embedding_names:
            embedding_api_key = os.environ.get('AZURE_OPENAI_APIKEY')
            embedding_api_version = cs.get_string('LLMS.AZURE.OPENAI', 'embedding_api_version')
            # Append embedding model name to base endpoint
            embedding_endpoint = cs.get_string('LLMS.AZURE.OPENAI', 'embedding_endpoint') + embedding
        
        if api_key is None or model_endpoint is None:
            raise ValueError(
                f"Azure API key or endpoint not found for model '{model}'. "
                f"AI models: {self._modelnames_ai}, OpenAI models: {self._modelnames_openai}"
            )
        
        logging.info("Azure model endpoint: %s", model_endpoint)
        logging.info("Azure embedding endpoint: %s", embedding_endpoint)
        
        return (api_key, model_api_version, model_endpoint,
                embedding_api_key, embedding_api_version, embedding_endpoint)
    
    def create_chat_model(self, temperature: Optional[float] = None):
        """Create Azure chat model."""
        api_key, api_version, endpoint, _, _, _ = self._get_azure_settings()
        temp = temperature if temperature is not None else self.config.temperature
        model = self.config.model_text
        
        if model in self._modelnames_ai:
            logging.info("Model %s uses Azure AI API", model)
            return AzureAIChatCompletionsModel(
                endpoint=endpoint,
                credential=api_key,
                model_name=model
            )
        else:
            logging.info("Model %s uses Azure OpenAI API", model)
            return AzureAIChatCompletionsModel(
                endpoint=endpoint,
                credential=api_key,
                temperature=temp,
                model_name=model,
                max_completion_tokens=int(self.config.max_tokens),
                verbose=True,
                api_version=api_version,
                client_kwargs={"logging_enable": True}
            )
    
    def create_embeddings(self):
        """Create Azure embeddings."""
        _, _, _, api_key, api_version, endpoint = self._get_azure_settings()
        
        return AzureAIEmbeddingsModel(
            endpoint=endpoint,
            credential=api_key,
            model_name=self.config.embedding_model,
            api_version=api_version
        )
