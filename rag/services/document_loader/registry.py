"""Registry for document loader strategies."""

import logging
from typing import Dict, Any
from langchain_chroma import Chroma
from .pdf_loader import PDFLoaderStrategy
from .text_loader import TextLoaderStrategy
from .html_loader import HTMLLoaderStrategy
from .word_loader import WordLoaderStrategy
from .powerpoint_loader import PowerPointLoaderStrategy
from .excel_loader import ExcelLoaderStrategy


class DocumentLoaderRegistry:
    """Registry for managing document loader strategies."""
    
    def __init__(self):
        """Initialize the registry with default loaders."""
        self._loaders = {
            'pdf': PDFLoaderStrategy(),
            'txt': TextLoaderStrategy(),
            'html': HTMLLoaderStrategy(),
            'docx': WordLoaderStrategy(),
            'pptx': PowerPointLoaderStrategy(),
            'xlsx': ExcelLoaderStrategy(),
        }
    
    def load_documents(
        self, 
        file_type: str, 
        config: Dict[str, Any], 
        vectorstore: Chroma
    ) -> int:
        """
        Load documents of specified type into vector store.
        
        Args:
            file_type: Type of files to load ('pdf', 'txt', etc.) or 'all'
            config: Configuration dictionary
            vectorstore: Vector store instance
            
        Returns:
            Total number of document splits loaded
        """
        if file_type == 'all':
            total = 0
            for loader_type, loader in self._loaders.items():
                try:
                    loader_config = {**config, 'glob': f'**/*.{loader_type}'}
                    count = loader.load(loader_config, vectorstore)
                    total += count
                except Exception as e:
                    logging.error("Error loading %s files: %s", loader_type, e)
            return total
        
        loader = self._loaders.get(file_type)
        if not loader:
            raise ValueError(f"Unknown file type: {file_type}")
        
        return loader.load(config, vectorstore)
    
    def register_loader(self, file_type: str, loader) -> None:
        """Register a new loader strategy."""
        self._loaders[file_type] = loader
    
    def list_supported_types(self) -> list:
        """Get list of supported file types."""
        return list(self._loaders.keys())
