#!/usr/bin/env python3
"""CLI tool to fetch X (Twitter) posts and save as text files.

Usage:
    python scripts/fetch_x_post.py <x_url> [-o output_dir]
    python scripts/fetch_x_post.py --file urls.txt [-o output_dir]

The script reads X_API_KEY from env/config.env (or environment variable).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag.services.document_loader.x_loader import XLoaderStrategy

# Load environment from env/config.env
load_dotenv('env/config.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_and_save_post(url: str, output_dir: Path) -> bool:
    """Fetch a single X post and save to file.
    
    Args:
        url: X post URL
        output_dir: Directory to save the text file
        
    Returns:
        True if successful, False otherwise
    """
    loader = XLoaderStrategy()
    
    # Extract tweet ID
    tweet_id = loader._extract_tweet_id(url)
    if not tweet_id:
        logger.error(f"Could not extract tweet ID from: {url}")
        return False
    
    # Fetch tweet data
    tweet_data = loader._fetch_tweet(tweet_id)
    if not tweet_data:
        logger.error(f"Failed to fetch tweet: {url}")
        return False
    
    # Extract data
    tweet = tweet_data.get('data', {})
    includes = tweet_data.get('includes', {}).get('users', [{}])[0] if tweet_data.get('includes', {}).get('users') else {}
    
    author_username = includes.get('username', 'unknown')
    author_name = includes.get('name', 'Unknown')
    text = tweet.get('text', '')
    tweet_id = tweet.get('id', '')
    created_at = tweet.get('created_at', '')
    
    # Generate title from first line or truncate
    first_line = text.split('\n')[0] if text else ''
    if len(first_line) > 50:
        title = first_line[:47] + '...'
    else:
        title = first_line
    
    # Generate summary (first 200 chars of tweet)
    if len(text) > 200:
        summary = text[:197] + '...'
    else:
        summary = text
    
    # Build output content
    content = f"""Author: @{author_username} ({author_name})
Title: {title}
Posted: {created_at}
URL: https://x.com/{author_username}/status/{tweet_id}

Summary:
{summary}

---

Full Post:
{text}
"""
    
    # Create safe filename
    safe_username = ''.join(c for c in author_username if c.isalnum() or c in '-_')
    filename = f"{safe_username}_{tweet_id}.txt"
    output_path = output_dir / filename
    
    # Write to file
    output_path.write_text(content, encoding='utf-8')
    logger.info(f"Saved to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fetch X (Twitter) posts and save as text files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'urls',
        nargs='*',
        help='X post URLs to fetch'
    )
    parser.add_argument(
        '-f', '--file',
        help='File containing URLs (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/x_exports',
        help='Output directory (default: data/x_exports)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect URLs
    urls = []
    
    if args.file:
        urls_file = Path(args.file)
        if urls_file.exists():
            urls = [line.strip() for line in urls_file.read_text().split('\n') if line.strip()]
        else:
            logger.error(f"URLs file not found: {args.file}")
            sys.exit(1)
    elif args.urls:
        urls = args.urls
    else:
        parser.print_help()
        sys.exit(1)
    
    # Check API key (loaded from env/config.env)
    api_key = os.environ.get('X_API_KEY') or os.environ.get('TWITTER_BEARER_TOKEN')
    if not api_key:
        logger.error("X_API_KEY not found in env/config.env or environment")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fetching {len(urls)} X posts...")
    
    success_count = 0
    fail_count = 0
    
    for url in urls:
        if fetch_and_save_post(url, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Done: {success_count} successful, {fail_count} failed")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
