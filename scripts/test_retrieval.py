#!/usr/bin/env python3
"""Test retrieval from vector store."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env vars
from dotenv import load_dotenv
load_dotenv('/home/leen/projects/RAG/env/config.env')

import logging
logging.basicConfig(level=logging.INFO)

from rag.config.settings import ConfigService
from rag.models.config_models import RAGConfig
from rag.services.embeddings import EmbeddingsService
from rag.services.vector_store import VectorStoreService


def main():
    config_service = ConfigService('constants/constants_nebul.ini')
    rag_config = RAGConfig.from_config_file(config_service, 'nebul')

    print(f"Config score threshold: {rag_config.score}")
    print(f"Config similar k: {rag_config.similar}")
    print()

    embeddings_service = EmbeddingsService(rag_config, config_service=config_service)
    vector_store_service = VectorStoreService(rag_config, embeddings_service)

    # Try queries
    queries = [
        'AI machine learning',
        'GPT LLM',
        'neural network',
    ]

    for query in queries:
        print(f"\n=== Query: '{query}' ===")
        results = vector_store_service.search_similar(query, k=5)
        
        print(f"Results found: {len(results)}")
        
        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i+1}. [score={score:.4f}] source={source}")
            print(f"     {preview}...")


if __name__ == '__main__':
    main()
