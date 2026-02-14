"""Word document loader strategy."""

import logging
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class WordLoaderStrategy:
    """Strategy for loading Word (.docx) documents."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load Word documents into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        loader_kwargs = {'autodetect_encoding': True, 'mode': 'single'}
        loader = DirectoryLoader(
            path=config['data_dir'],
            glob=config['glob'],
            loader_cls=UnstructuredWordDocumentLoader,
            silent_errors=False,
            loader_kwargs=loader_kwargs
        )
        
        try:
            docs = loader.load()
            if docs:
                if docs[0].page_content == '':
                    logging.error("No content in %s", docs[0].metadata.get('source', 'unknown'))
                    return 0
                # Clean metadata (convert lists to comma-separated strings)
                for doc in docs:
                    if doc.metadata:
                        for key in doc.metadata:
                            if isinstance(doc.metadata[key], list):
                                doc.metadata[key] = ','.join([str(item) for item in doc.metadata[key]])
                vectorstore.add_documents(docs)
                logging.info("Loaded %d Word documents", len(docs))
                return len(docs)
        except Exception as e:
            logging.error("Error loading Word documents: %s", e)
        
        return 0
