"""X (Twitter) document loader strategy.

This loader fetches X posts via the X API v2 and converts them into
documents for vectorization.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class XLoaderStrategy:
    """Strategy for loading X (Twitter) posts into vector store."""
    
    # X URL patterns to match
    URL_PATTERNS = [
        r'twitter\.com/\w+/status/\d+',
        r'x\.com/\w+/status/\d+',
    ]
    
    def __init__(self):
        """Initialize the X loader strategy."""
        self._api_token: Optional[str] = None
        self._api_base_url = "https://api.twitter.com/2/tweets"
    
    @property
    def api_token(self) -> Optional[str]:
        """Get the X API token from environment or config."""
        if self._api_token is None:
            # Try to get from environment variable
            self._api_token = os.environ.get('X_API_KEY') or os.environ.get('TWITTER_BEARER_TOKEN')
        return self._api_token
    
    @api_token.setter
    def api_token(self, value: str):
        """Set the X API token."""
        self._api_token = value
    
    def _extract_tweet_id(self, url: str) -> Optional[str]:
        """Extract tweet ID from X URL.
        
        Args:
            url: X/Twitter URL
            
        Returns:
            Tweet ID if found, None otherwise
        """
        # Match twitter.com/user/status/ID or x.com/user/status/ID
        pattern = r'(?:twitter|x)\.com/\w+/status/(\d+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None
    
    def _is_x_url(self, url: str) -> bool:
        """Check if URL is a valid X/Twitter post URL."""
        return any(re.search(pattern, url, re.IGNORECASE) 
                   for pattern in self.URL_PATTERNS)
    
    def _fetch_tweet(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Fetch tweet data from X API v2.
        
        Args:
            tweet_id: The tweet ID to fetch
            
        Returns:
            Tweet data dict or None if failed
        """
        import requests
        
        if not self.api_token:
            logging.error("X API token not configured. Set X_API_KEY environment variable.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,text,source",
            "expansions": "author_id",
            "user.fields": "username,name,description"
        }
        
        try:
            response = requests.get(
                f"{self._api_base_url}/{tweet_id}",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logging.error("X API authentication failed. Check your API key.")
            elif response.status_code == 404:
                logging.error(f"Tweet {tweet_id} not found.")
            else:
                logging.error(f"X API error: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            logging.error(f"Error fetching tweet from X API: {e}")
        
        return None
    
    def _tweet_to_document(self, tweet_data: Dict[str, Any]) -> Document:
        """Convert tweet data to a LangChain Document.
        
        Args:
            tweet_data: Tweet data from X API
            
        Returns:
            LangChain Document
        """
        tweet = tweet_data.get('data', {})
        includes = tweet_data.get('includes', {}).get('users', [{}])[0] if tweet_data.get('includes', {}).get('users') else {}
        
        # Build text content
        text_parts = []
        
        # Author info
        author_name = includes.get('name', 'Unknown')
        author_username = includes.get('username', 'unknown')
        text_parts.append(f"@{author_username} ({author_name})")
        
        # Tweet text
        text_parts.append(tweet.get('text', ''))
        
        # Metrics
        metrics = tweet.get('public_metrics', {})
        if metrics:
            metrics_str = (
                f"Retweets: {metrics.get('retweet_count', 0)}, "
                f"Likes: {metrics.get('like_count', 0)}, "
                f"Replies: {metrics.get('reply_count', 0)}, "
                f"Quotes: {metrics.get('quote_count', 0)}"
            )
            text_parts.append(metrics_str)
        
        # Timestamp
        created_at = tweet.get('created_at', '')
        if created_at:
            text_parts.append(f"Posted: {created_at}")
        
        # Build metadata
        metadata = {
            'source': 'x',
            'tweet_id': tweet.get('id'),
            'author_id': tweet.get('author_id'),
            'author_username': author_username,
            'author_name': author_name,
            'created_at': created_at,
            'retweet_count': metrics.get('retweet_count', 0),
            'like_count': metrics.get('like_count', 0),
            'reply_count': metrics.get('reply_count', 0),
            'quote_count': metrics.get('quote_count', 0)
        }
        
        content = '\n'.join(text_parts)
        return Document(page_content=content, metadata=metadata)
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load X posts into vector store.
        
        Args:
            config: Configuration dictionary containing:
                - x_urls: List of X URLs to load
                - chunk_size: Size of text chunks
                - chunk_overlap: Overlap between chunks
            vectorstore: Vector store to add documents to
            
        Returns:
            Number of document splits loaded
        """
        x_urls = config.get('x_urls', [])
        
        if not x_urls:
            logging.info("No X URLs provided to load")
            return 0
        
        # Filter valid X URLs
        valid_urls = [url for url in x_urls if self._is_x_url(url)]
        
        if not valid_urls:
            logging.warning("No valid X URLs found in provided list")
            return 0
        
        logging.info(f"Loading {len(valid_urls)} X posts")
        
        # Text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 50)
        )
        
        documents = []
        
        for url in valid_urls:
            tweet_id = self._extract_tweet_id(url)
            if not tweet_id:
                logging.warning(f"Could not extract tweet ID from URL: {url}")
                continue
            
            tweet_data = self._fetch_tweet(tweet_id)
            if not tweet_data:
                logging.warning(f"Failed to fetch tweet: {url}")
                continue
            
            try:
                doc = self._tweet_to_document(tweet_data)
                documents.append(doc)
                logging.info(f"Loaded tweet {tweet_id} from {url}")
            except Exception as e:
                logging.error(f"Error processing tweet {tweet_id}: {e}")
        
        if not documents:
            logging.warning("No documents loaded from X URLs")
            return 0
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        if splits:
            vectorstore.add_documents(splits)
            logging.info(f"Loaded {len(splits)} X document splits into vector store")
            return len(splits)
        
        return 0
    
    def load_single(self, url: str, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load a single X post into vector store.
        
        This is a convenience method for loading a single URL via API.
        
        Args:
            url: X URL to load
            config: Configuration dictionary
            vectorstore: Vector store to add documents to
            
        Returns:
            Number of document splits loaded (0 or 1)
        """
        return self.load({**config, 'x_urls': [url]}, vectorstore)


# Singleton instance for reuse
_x_loader_instance: Optional[XLoaderStrategy] = None


def get_x_loader() -> XLoaderStrategy:
    """Get the singleton X loader instance."""
    global _x_loader_instance
    if _x_loader_instance is None:
        _x_loader_instance = XLoaderStrategy()
    return _x_loader_instance
