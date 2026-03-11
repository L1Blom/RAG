"""X (Twitter) document loader strategy.

This loader fetches X posts via the X API v2 and converts them into
documents for vectorization.
"""

import logging
import os
import re
import json
import tempfile
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata


class XLoaderError(Exception):
    """Exception raised for X loader errors."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


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
        # Strip query parameters before matching (e.g., ?s=46)
        url_without_params = url.split('?')[0] if '?' in url else url
        pattern = r'(?:twitter|x)\.com/\w+/status/(\d+)'
        match = re.search(pattern, url_without_params)
        if match:
            return match.group(1)
        return None
    
    def _is_x_url(self, url: str) -> bool:
        """Check if URL is a valid X/Twitter post URL."""
        # Strip query parameters before matching
        url_without_params = url.split('?')[0] if '?' in url else url
        return any(re.search(pattern, url_without_params, re.IGNORECASE) 
                   for pattern in self.URL_PATTERNS)
    
    def _fetch_tweet(self, tweet_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch tweet data from X API v2.
        
        Args:
            tweet_id: The tweet ID to fetch
            max_retries: Maximum number of retry attempts for rate limits
            
        Returns:
            Tweet data dict or None if failed
        """
        import time
        
        if not self.api_token:
            logging.error("X API token not configured. Set X_API_KEY environment variable.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,text,source,attachments",
            "expansions": "author_id,attachments.media_keys",
            "user.fields": "username,name,description",
            "media.fields": "type,url,preview_image_url,variants,duration_ms"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self._api_base_url}/{tweet_id}",
                    headers=headers,
                    params=params,
                    timeout=30
                )
                
                # Debug: log response status and first part of response
                logging.debug(f"X API response status: {response.status_code}")
                logging.debug(f"X API response headers: {response.headers.get('content-type')}")
                
                if response.status_code == 200:
                    data = response.json()
                    # Check if response contains actual tweet data or error
                    if 'data' not in data:
                        logging.error(f"X API returned 200 but no 'data' field: {str(data)[:500]}")
                        return None
                    return data
                elif response.status_code == 401:
                    logging.error("X API authentication failed. Check your API key.")
                    return None
                elif response.status_code == 404:
                    logging.error(f"Tweet {tweet_id} not found.")
                    return None
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logging.warning(f"X API rate limit exceeded. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"X API error: {response.status_code} - {response.text[:500]}")
                    return None
                    
            except requests.RequestException as e:
                logging.error(f"Error fetching tweet from X API: {e}")
                return None
        
        # All retries exhausted
        logging.error(f"Failed to fetch tweet {tweet_id} after {max_retries} attempts due to rate limiting")
        return None
    
    def _tweet_to_document(self, tweet_data: Dict[str, Any], url: str = '') -> Document:
        """Convert tweet data to a LangChain Document.
        
        Args:
            tweet_data: Tweet data from X API
            url: Original X URL (optional)
            
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
        
        # X post link
        post_url = url if url else f"https://x.com/{author_username}/status/{tweet.get('id', '')}"
        text_parts.append(f"Original post: {post_url}")
        
        # Build metadata
        metadata = {
            'source': 'x',
            'tweet_id': tweet.get('id'),
            'author_id': tweet.get('author_id'),
            'author_username': author_username,
            'author_name': author_name,
            'created_at': created_at,
            'post_url': post_url,
            'retweet_count': metrics.get('retweet_count', 0),
            'like_count': metrics.get('like_count', 0),
            'reply_count': metrics.get('reply_count', 0),
            'quote_count': metrics.get('quote_count', 0)
        }
        
        content = '\n'.join(text_parts)
        return Document(page_content=content, metadata=metadata)

    def _tweet_to_text(self, tweet_data: Dict[str, Any], url: str = '') -> str:
        """Convert tweet data to normalized plain text for post.txt."""
        return self._tweet_to_document(tweet_data, url).page_content

    def _build_post_dir(self, data_dir: str, tweet_id: str) -> str:
        """Build and create the canonical storage directory for one X post."""
        post_dir = os.path.join(data_dir, str(tweet_id))
        os.makedirs(post_dir, exist_ok=True)
        return post_dir

    def _ensure_media_dirs(self, post_dir: str) -> Dict[str, str]:
        """Ensure image/video/audio subdirectories exist for a post."""
        dirs = {
            'images': os.path.join(post_dir, 'images'),
            'videos': os.path.join(post_dir, 'videos'),
            'audio': os.path.join(post_dir, 'audio'),
        }
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        return dirs

    def _extract_media_urls(self, tweet_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract image/video/audio URLs from tweet media includes."""
        media = tweet_data.get('includes', {}).get('media', [])
        result: Dict[str, List[str]] = {'images': [], 'videos': [], 'audio': []}

        for item in media:
            media_type = item.get('type', '')
            if media_type == 'photo' and item.get('url'):
                result['images'].append(item['url'])
                continue

            variants = item.get('variants', []) or []
            video_variants = [v for v in variants if str(v.get('content_type', '')).startswith('video/') and v.get('url')]
            audio_variants = [v for v in variants if str(v.get('content_type', '')).startswith('audio/') and v.get('url')]

            if video_variants:
                best_video = max(video_variants, key=lambda v: int(v.get('bit_rate') or 0))
                result['videos'].append(best_video['url'])
            if audio_variants:
                best_audio = max(audio_variants, key=lambda v: int(v.get('bit_rate') or 0))
                result['audio'].append(best_audio['url'])

        return result

    def _download_asset(self, url: str, target_path: str, timeout: int = 30, max_bytes: int = 50 * 1024 * 1024) -> bool:
        """Download one media asset to disk."""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code != 200:
                logging.warning("Failed downloading asset %s: HTTP %s", url, response.status_code)
                return False

            content_length = response.headers.get('Content-Length')
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        logging.warning("Skipping oversized asset %s (%s bytes)", url, content_length)
                        return False
                except ValueError:
                    pass

            base_dir = os.path.dirname(target_path)
            safe_target_path = self._safe_join(base_dir, os.path.basename(target_path))
            downloaded = 0

            with open(safe_target_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        logging.warning("Aborting oversized streamed asset %s", url)
                        file.close()
                        try:
                            os.remove(safe_target_path)
                        except OSError:
                            pass
                        return False
                    file.write(chunk)
            return True
        except requests.RequestException as exc:
            logging.warning("Failed downloading asset %s: %s", url, exc)
            return False
        except OSError as exc:
            logging.warning("Failed writing asset %s: %s", url, exc)
            return False

    def _persist_post_snapshot(self, post_dir: str, post_json: Dict[str, Any], post_text: str) -> None:
        """Persist post metadata and normalized text to local snapshot files."""
        post_json_path = os.path.join(post_dir, 'post.json')
        post_txt_path = os.path.join(post_dir, 'post.txt')

        # Atomic write for JSON snapshot
        with tempfile.NamedTemporaryFile('w', dir=post_dir, delete=False, prefix='post_json_', suffix='.tmp') as tmp_json:
            json.dump(post_json, tmp_json, indent=2)
            tmp_json_path = tmp_json.name
        os.replace(tmp_json_path, post_json_path)

        # Atomic write for text snapshot
        with tempfile.NamedTemporaryFile('w', dir=post_dir, delete=False, prefix='post_txt_', suffix='.tmp') as tmp_txt:
            tmp_txt.write(post_text)
            tmp_txt_path = tmp_txt.name
        os.replace(tmp_txt_path, post_txt_path)

    def _safe_join(self, base_dir: str, *parts: str) -> str:
        """Safely join paths and prevent path traversal outside base_dir."""
        base_real = os.path.realpath(base_dir)
        candidate = os.path.realpath(os.path.join(base_dir, *parts))
        if os.path.commonpath([base_real, candidate]) != base_real:
            raise ValueError(f"Unsafe path detected: {candidate}")
        return candidate

    def _sanitize_extension(self, extension: str) -> str:
        """Sanitize file extension derived from URL path."""
        if not extension:
            return '.bin'
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        ext = re.sub(r'[^a-z0-9.]', '', ext)
        if len(ext) > 10:
            ext = ext[:10]
        if ext in ('', '.'):
            return '.bin'
        return ext
    
    def load(self, config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Load X posts into vector store.
        
        Loads from x.json file in data directory, or from x_urls config key.
        
        Args:
            config: Configuration dictionary containing:
                - data_dir: Directory containing x.json (optional)
                - x_urls: List of X URLs to load (optional, overrides x.json)
                - chunk_size: Size of text chunks
                - chunk_overlap: Overlap between chunks
            vectorstore: Vector store to add documents to
            
        Returns:
            Number of document splits loaded
        """
        # When no x_urls are provided, reload from local snapshots only.
        x_urls = config.get('x_urls', [])
        if not x_urls:
            documents = self._load_local_snapshot_documents(config.get('data_dir', ''))
            return self._split_and_store_documents(documents, config, vectorstore)
        
        # Filter valid X URLs
        valid_urls = [url for url in x_urls if self._is_x_url(url)]
        
        if not valid_urls:
            logging.warning("No valid X URLs found in provided list")
            return 0
        
        logging.info(f"Loading {len(valid_urls)} X posts")
        
        documents = []
        data_dir = config.get('data_dir', '')
        download_media = bool(config.get('download_media', False))
        download_timeout = int(config.get('download_timeout', 30))
        max_download_mb = int(config.get('max_download_mb', 50))
        
        for url in valid_urls:
            tweet_id = self._extract_tweet_id(url)
            if not tweet_id:
                logging.warning(f"Could not extract tweet ID from URL: {url}")
                continue
            
            tweet_data = self._fetch_tweet(tweet_id)
            if not tweet_data:
                logging.warning(f"Failed to fetch tweet: {url} (ID: {tweet_id})")
                continue
            
            try:
                post_text = self._tweet_to_text(tweet_data, url)
                author_users = tweet_data.get('includes', {}).get('users') or [{}]
                author_user = author_users[0]
                media_urls = self._extract_media_urls(tweet_data)
                images_count = len(media_urls.get('images', []))
                videos_count = len(media_urls.get('videos', []))
                audio_count = len(media_urls.get('audio', []))

                snapshot_path = ''
                text_path = ''

                # Persist local snapshot for upload/reload workflows.
                if data_dir:
                    post_dir = self._build_post_dir(data_dir, tweet_id)
                    media_dirs = self._ensure_media_dirs(post_dir)
                    assets = {'images': [], 'videos': [], 'audio': []}

                    for media_type, urls in media_urls.items():
                        for idx, media_url in enumerate(urls, start=1):
                            parsed = urlparse(media_url)
                            ext = self._sanitize_extension(os.path.splitext(parsed.path)[1])
                            prefix = {'images': 'img', 'videos': 'vid', 'audio': 'aud'}[media_type]
                            filename = f"{prefix}_{idx:02d}{ext}"
                            target_path = self._safe_join(media_dirs[media_type], filename)

                            status = 'skipped'
                            if download_media:
                                status = 'downloaded' if self._download_asset(
                                    media_url,
                                    target_path,
                                    timeout=download_timeout,
                                    max_bytes=max_download_mb * 1024 * 1024,
                                ) else 'failed'

                            assets[media_type].append({
                                'url': media_url,
                                'local_path': target_path,
                                'status': status,
                            })

                    post_json = {
                        'tweet_id': tweet_id,
                        'post_url': url,
                        'text': tweet_data.get('data', {}).get('text', ''),
                        'created_at': tweet_data.get('data', {}).get('created_at', ''),
                        'author': {
                            'id': tweet_data.get('data', {}).get('author_id', ''),
                            'username': author_user.get('username', 'unknown'),
                            'name': author_user.get('name', 'Unknown'),
                        },
                        'assets': assets,
                        'indexing': {
                            'mode': 'text_only',
                            'video_indexed': False,
                            'audio_indexed': False,
                        },
                        'errors': [],
                    }
                    self._persist_post_snapshot(post_dir, post_json, post_text)
                    snapshot_path = os.path.join(post_dir, 'post.json')
                    text_path = os.path.join(post_dir, 'post.txt')

                metadata = {
                    'source': 'x',
                    'tweet_id': tweet_data.get('data', {}).get('id') or tweet_id,
                    'author_id': tweet_data.get('data', {}).get('author_id'),
                    'author_username': author_user.get('username', 'unknown'),
                    'author_name': author_user.get('name', 'Unknown'),
                    'created_at': tweet_data.get('data', {}).get('created_at', ''),
                    'post_url': url,
                    'snapshot_path': snapshot_path,
                    'text_path': text_path,
                    'images_count': images_count,
                    'videos_count': videos_count,
                    'audio_count': audio_count,
                    'indexing_mode': 'text_only',
                }
                doc = Document(page_content=post_text, metadata=metadata)
                documents.append(doc)
                logging.info(f"Loaded tweet {tweet_id} from {url}")
            except Exception as e:
                logging.error(f"Error processing tweet {tweet_id}: {e}")
        
        if not documents:
            logging.warning("No documents loaded from X URLs")
            return 0

        return self._split_and_store_documents(documents, config, vectorstore)

    def _split_and_store_documents(self, documents: List[Document], config: Dict[str, Any], vectorstore: Chroma) -> int:
        """Split and store documents in the vectorstore."""
        if not documents:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 50)
        )

        splits = text_splitter.split_documents(documents)
        if not splits:
            return 0

        splits = filter_complex_metadata(splits)
        vectorstore.add_documents(splits)
        logging.info("Loaded %d X document splits into vector store", len(splits))
        return len(splits)

    def _load_local_snapshot_documents(self, data_dir: str) -> List[Document]:
        """Load X post documents from local snapshots under data/<tweet_id>."""
        if not data_dir or not os.path.isdir(data_dir):
            logging.info("No local X snapshot directory found at: %s", data_dir)
            return []

        documents: List[Document] = []
        for entry in os.listdir(data_dir):
            post_dir = os.path.join(data_dir, entry)
            if not os.path.isdir(post_dir):
                continue

            post_json_path = os.path.join(post_dir, 'post.json')
            post_txt_path = os.path.join(post_dir, 'post.txt')
            if not os.path.exists(post_json_path):
                continue

            try:
                with open(post_json_path, 'r') as file:
                    post_data = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Skipping invalid X snapshot %s: %s", post_json_path, exc)
                continue

            post_text = ''
            if os.path.exists(post_txt_path):
                try:
                    with open(post_txt_path, 'r') as file:
                        post_text = file.read()
                except OSError as exc:
                    logging.warning("Could not read snapshot text %s: %s", post_txt_path, exc)
            if not post_text:
                post_text = str(post_data.get('text', '')).strip()
            if not post_text:
                continue

            assets = post_data.get('assets', {})
            author = post_data.get('author', {})
            metadata = {
                'source': 'x',
                'tweet_id': post_data.get('tweet_id', entry),
                'post_url': post_data.get('post_url', ''),
                'author_username': author.get('username', post_data.get('author_username', 'unknown')),
                'created_at': post_data.get('created_at', ''),
                'snapshot_path': post_json_path,
                'text_path': post_txt_path if os.path.exists(post_txt_path) else '',
                'images_count': len(assets.get('images', [])),
                'videos_count': len(assets.get('videos', [])),
                'audio_count': len(assets.get('audio', [])),
                'indexing_mode': 'text_only',
            }
            documents.append(Document(page_content=post_text, metadata=metadata))

        logging.info("Loaded %d local X snapshot documents", len(documents))
        return documents
    
    def _load_x_urls(self, data_dir: str) -> Optional[List[str]]:
        """Load X URLs from x.json file in data directory.
        
        Args:
            data_dir: Directory containing x.json
            
        Returns:
            List of X URLs or None if file not found
        """
        x_urls_file = os.path.join(data_dir, 'x.json')
        if not os.path.exists(x_urls_file):
            return None
        
        try:
            with open(x_urls_file, 'r') as file:
                contents = file.read()
                if contents.strip() == "":
                    return []
                contents = json.loads(contents)
                if not isinstance(contents, list):
                    logging.error("x.json file does not contain a list")
                    return None
                logging.info("Found %d X URLs in x.json", len(contents))
                return contents
        except json.JSONDecodeError as e:
            logging.error("Error reading x.json file: %s", e)
            return None

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
