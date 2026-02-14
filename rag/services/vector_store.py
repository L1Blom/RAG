"""Vector store management service."""

import logging
from typing import Optional
from langchain_chroma import Chroma
import chromadb
from rag.models.config_models import RAGConfig
from rag.services.embeddings import EmbeddingsService
from rag.services.document_loader.registry import DocumentLoaderRegistry


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self, config: RAGConfig, embeddings_service: EmbeddingsService):
        """
        Initialize vector store service.
        
        Args:
            config: RAG configuration instance
            embeddings_service: Embeddings service instance
        """
        self.config = config
        self.embeddings_service = embeddings_service
        self._vectorstore: Optional[Chroma] = None
        self._document_loader = DocumentLoaderRegistry()
    
    @property
    def vectorstore(self) -> Chroma:
        """Get vector store instance (lazy loaded)."""
        if self._vectorstore is None:
            self._vectorstore = self._create_vectorstore()
        return self._vectorstore
    
    def _create_vectorstore(self) -> Chroma:
        """Create or load vector store."""
        logging.info("Initializing vector store at: %s", self.config.persistence_dir)
        
        vectorstore = Chroma(
            persist_directory=self.config.persistence_dir,
            embedding_function=self.embeddings_service.embeddings
        )
        
        # Check if vector store has existing data
        collection = vectorstore._collection
        count = collection.count()
        
        if count > 0:
            logging.info("Loaded %d chunks from persistent vectorstore", count)
        else:
            logging.info("Vector store is empty, loading documents...")
            # Pass the vectorstore directly to avoid recursion via the property
            self._reload_into(vectorstore)
        
        return vectorstore
    
    def reload_documents(self, file_type: str = 'all') -> int:
        """
        Reload documents into vector store.
        
        Args:
            file_type: Type of files to load or 'all'
            
        Returns:
            Number of document splits loaded
        """
        return self._reload_into(self.vectorstore, file_type)
    
    def _reload_into(self, vectorstore: Chroma, file_type: str = 'all') -> int:
        """
        Internal helper – load documents into the given vectorstore instance.
        
        Args:
            vectorstore: The Chroma vectorstore to load documents into
            file_type: Type of files to load or 'all'
            
        Returns:
            Number of document splits loaded
        """
        logging.info("Reloading documents of type: %s", file_type)
        
        config_dict = {
            'data_dir': self.config.data_dir,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'glob': f'*.{file_type}' if file_type != 'all' else '*.*'
        }
        
        count = self._document_loader.load_documents(
            file_type,
            config_dict,
            vectorstore
        )
        
        logging.info("Loaded %d document splits", count)
        return count
    
    def search_similar(self, query: str, k: Optional[int] = None) -> list:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return (uses config.similar if not specified)
            
        Returns:
            List of similar documents with scores
        """
        k = k or self.config.similar
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def reset(self) -> None:
        """Reset vector store instance."""
        self._vectorstore = None
