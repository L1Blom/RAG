"""Base protocol for document loader strategies."""

from typing import Protocol, Dict, Any
from langchain_chroma import Chroma


class DocumentLoaderStrategy(Protocol):
    """Protocol for document loader strategies."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """
        Load documents into the vector store.
        
        Args:
            config: Configuration dictionary containing:
                - data_dir: Directory containing documents
                - glob: File pattern to match
                - chunk_size: Size of text chunks
                - chunk_overlap: Overlap between chunks
            vectorstore: Vector store to add documents to
            
        Returns:
            Number of document splits loaded
        """
        ...
