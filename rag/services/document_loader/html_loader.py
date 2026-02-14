"""HTML document loader strategy."""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HTMLLoaderStrategy:
    """Strategy for loading HTML documents from URLs."""
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load HTML documents from URLs into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        urls = self._load_urls(config.get('data_dir', ''))
        if not urls:
            logging.info("No URLs found for HTML loading")
            return 0
        
        try:
            loader = WebBaseLoader(
                web_paths=urls,
                continue_on_failure=True,
            )
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            if splits:
                vectorstore.add_documents(splits)
                logging.info("Loaded %d HTML document splits from %d URLs", len(splits), len(urls))
                return len(splits)
        except Exception as e:
            logging.error("Error loading HTML documents: %s", e)
        
        return 0
    
    def _load_urls(self, data_dir: str) -> Optional[List[str]]:
        """Load URLs from urls.json file in data directory."""
        urls_file = os.path.join(data_dir, 'urls.json')
        if not os.path.exists(urls_file):
            return None
        
        try:
            with open(urls_file, 'r') as file:
                contents = file.read()
                if contents.strip() == "":
                    return []
                contents = json.loads(contents)
                if not isinstance(contents, list):
                    logging.error("URLs file does not contain a list")
                    return None
                logging.info("Found %d URLs in file", len(contents))
                return contents
        except json.JSONDecodeError as e:
            logging.error("Error reading URLs file: %s", e)
            return None
