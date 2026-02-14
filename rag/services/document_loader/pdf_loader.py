"""PDF document loader strategy."""

import logging
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFLoaderStrategy:
    """Strategy for loading PDF documents."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load PDF documents into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        loader = PyPDFDirectoryLoader(
            path=config['data_dir'],
            glob=config['glob']
        )
        
        try:
            splits = loader.load_and_split()
            if splits:
                vectorstore.add_documents(splits)
                logging.info("Loaded %d PDF document splits", len(splits))
                return len(splits)
        except Exception as e:
            logging.error("Error loading PDF documents: %s", e)
        
        return 0
