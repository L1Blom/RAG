"""Text document loader strategy."""

import logging
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class TextLoaderStrategy:
    """Strategy for loading text documents."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load text documents into vector store."""
        # Support optional language-aware splitting
        language = config.get('language')
        if language:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language[language],
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
        
        loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(
            path=config['data_dir'],
            glob=config['glob'],
            loader_cls=TextLoader,
            silent_errors=True,
            loader_kwargs=loader_kwargs
        )
        
        try:
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            if splits:
                vectorstore.add_documents(splits)
                logging.info("Loaded %d text document splits", len(splits))
                return len(splits)
        except Exception as e:
            logging.error("Error loading text documents: %s", e)
        
        return 0
