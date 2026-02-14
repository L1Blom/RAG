"""Excel document loader strategy."""

import logging
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredExcelLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ExcelLoaderStrategy:
    """Strategy for loading Excel (.xlsx) documents."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load Excel documents into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        loader_kwargs = {'mode': 'elements'}
        loader = DirectoryLoader(
            path=config['data_dir'],
            glob=config['glob'],
            loader_cls=UnstructuredExcelLoader,
            silent_errors=True,
            loader_kwargs=loader_kwargs
        )
        
        try:
            docs = loader.load()
            # Filter out empty elements and clean metadata
            cleaned_docs = []
            for doc in docs:
                if not doc.page_content.strip():
                    continue
                if doc.metadata:
                    for key in doc.metadata:
                        if isinstance(doc.metadata[key], list):
                            doc.metadata[key] = ','.join([str(item) for item in doc.metadata[key]])
                cleaned_docs.append(doc)
            
            if cleaned_docs:
                splits = text_splitter.split_documents(cleaned_docs)
                if splits:
                    vectorstore.add_documents(splits)
                    logging.info("Loaded %d Excel elements, %d splits", len(cleaned_docs), len(splits))
                    return len(splits)
        except Exception as e:
            logging.error("Error loading Excel documents: %s", e)
        
        return 0
