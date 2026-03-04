#!/usr/bin/env python3
"""Utility to inspect vector store contents.

Usage:
    python scripts/inspect_vectorstore.py <project>

Example:
    python scripts/inspect_vectorstore.py centric
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config.settings import ConfigService
from rag.models.config_models import RAGConfig
from rag.utils.debug_embeddings import inspect_vector_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project>")
        print("Example: python scripts/inspect_vectorstore.py centric")
        sys.exit(1)
    
    project = sys.argv[1]
    config_file = f"constants/constants_{project}.ini"
    
    # Load config
    config_service = ConfigService(config_file)
    rag_config = RAGConfig.from_config_file(config_service, project)
    
    # Inspect vector store
    logger.info(f"Project: {project}")
    logger.info(f"Data directory: {rag_config.data_dir}")
    logger.info(f"Persistence directory: {rag_config.persistence_dir}")
    logger.info("=" * 60)
    
    results = inspect_vector_store(rag_config.persistence_dir)
    
    logger.info("=" * 60)
    logger.info(f"Found {len(results)} documents in vector store")
    logger.info("=" * 60)
    
    # Print each document
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1}/{len(results)} ---")
        print(f"ID: {doc['id']}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Text preview: {doc['text_preview']}")
        print(f"Full text length: {len(doc['text'])} chars")
        
        if i >= 49 and len(results) > 50:
            print(f"\n... and {len(results) - 50} more documents")
            break


if __name__ == '__main__':
    main()
