"""HTML document loader strategy."""

import json
import logging
import os
import re
from typing import Dict, Any, Optional, List
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HTMLLoaderStrategy:
    """Strategy for loading HTML documents from URLs."""
    
    # URLs that should be skipped (need JavaScript or specialized API)
    SKIP_PATTERNS = [
        r'x\.com/',
        r'twitter\.com/',
    ]
    
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
        
        # Filter out URLs that should be handled by other loaders
        filtered_urls = [url for url in urls if not self._should_skip(url)]
        
        if len(filtered_urls) < len(urls):
            skipped = len(urls) - len(filtered_urls)
            logging.info("Skipped %d URLs (X/Twitter URLs - use x.json with X loader)", skipped)
        
        if not filtered_urls:
            logging.info("No valid HTML URLs to load after filtering")
            return 0
        
        try:
            loader = WebBaseLoader(
                web_paths=filtered_urls,
                continue_on_failure=True,
            )
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            if splits:
                vectorstore.add_documents(splits)
                logging.info("Loaded %d HTML document splits from %d URLs", len(splits), len(filtered_urls))
                return len(splits)
        except Exception as e:
            logging.error("Error loading HTML documents: %s", e)
        
        return 0
    
    def _should_skip(self, url: str) -> bool:
        """Check if URL should be skipped (handled by other loaders)."""
        return any(re.search(pattern, url, re.IGNORECASE) 
                   for pattern in self.SKIP_PATTERNS)
    
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
