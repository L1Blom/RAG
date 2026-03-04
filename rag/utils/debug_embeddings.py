"""Debug wrapper for embeddings to log what's being sent to the embedding model."""

import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def inspect_vector_store(persist_directory: str) -> List[dict]:
    """
    Inspect the contents of a Chroma vector store.
    
    Args:
        persist_directory: Path to the vector store directory
        
    Returns:
        List of dictionaries with 'id', 'text', 'metadata'
    """
    logger.info("Inspecting vector store at: %s", persist_directory)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=None  # We just want to read, not embed
    )
    
    collection = vectorstore._collection
    count = collection.count()
    
    logger.info("Total documents in vector store: %d", count)
    
    results = []
    
    # Get all documents in batches
    batch_size = 100
    for i in range(0, count, batch_size):
        batch = collection.get(
            ids=collection.get()['ids'][i:i+batch_size],
            include=['documents', 'metadatas']
        )
        
        for idx, doc_id in enumerate(batch['ids']):
            text = batch['documents'][idx] if batch['documents'] else ""
            metadata = batch['metadatas'][idx] if batch['metadatas'] else {}
            
            results.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata,
                'text_preview': text[:200] + "..." if len(text) > 200 else text
            })
    
    return results


class DebugEmbeddingsWrapper:
    """
    Wrapper around embeddings that logs all text being sent for embedding.
    
    This helps verify what content is actually being indexed in the vector DB.
    """
    
    def __init__(self, embeddings: Embeddings, log_level: int = logging.INFO):
        """
        Initialize debug wrapper.
        
        Args:
            embeddings: The actual embeddings instance to wrap
            log_level: Logging level (default: INFO)
        """
        self._embeddings = embeddings
        self._log_level = log_level
        self._embed_call_count = 0
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents and log what is being sent.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        self._embed_call_count += 1
        call_num = self._embed_call_count
        
        logger.log(
            self._log_level,
            "=== EMBEDDING CALL #%d: embed_documents (%d texts) ===",
            call_num,
            len(texts)
        )
        
        for i, text in enumerate(texts):
            # Log first 500 chars of each text to avoid huge logs
            preview = text[:500] if len(text) > 500 else text
            preview = preview.replace('\n', '\\n').replace('\r', '\\r')
            
            logger.log(
                self._log_level,
                "--- Document #%d (total chars: %d) ---\n%s",
                i,
                len(text),
                preview
            )
        
        # Actually call the embeddings
        result = self._embeddings.embed_documents(texts)
        
        logger.log(
            self._log_level,
            "=== END EMBEDDING CALL #%d: returned %d embeddings ===",
            call_num,
            len(result)
        )
        
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query and log what is being sent.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        logger.log(
            self._log_level,
            "=== EMBEDDING QUERY (chars: %d) ===\n%s",
            len(text),
            text[:500].replace('\n', '\\n')
        )
        
        result = self._embeddings.embed_query(text)
        
        logger.log(
            self._log_level,
            "=== END EMBEDDING QUERY (embedding dims: %d) ===",
            len(result)
        )
        
        return result
    
    def __getattr__(self, name: str):
        """Pass through any other attributes to the wrapped embeddings."""
        return getattr(self._embeddings, name)
