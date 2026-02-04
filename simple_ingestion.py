"""
RSS Feed Ingestion Module
Fetches articles from news feeds and normalizes them into a common format.
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional
import feedparser
import httpx
from dateutil import parser as date_parser
import structlog

logger = structlog.get_logger()


class Article:
    """Represents a single news article"""
    
    def __init__(self, url: str, title: str, published_at: datetime, 
                 source: str, category: str, snippet: str):
        self.url = url
        self.title = title
        self.published_at = published_at
        self.source = source
        self.category = category
        self.snippet = snippet
        # Generate unique ID from URL
        self.id = hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def __repr__(self):
        return f"Article({self.source}: {self.title[:50]}...)"


async def fetch_feed(url: str, source_name: str, category: str, 
                     timeout: int = 30) -> List[Article]:
    """
    Fetch and parse a single RSS feed.
    
    Args:
        url: RSS feed URL
        source_name: Name of the news source
        category: Category (cybersecurity, technology, crypto)
        timeout: Request timeout in seconds
    
    Returns:
        List of Article objects
    """
    articles = []
    
    try:
        # Async HTTP request to fetch RSS feed
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("fetching_feed", source=source_name, url=url)
            response = await client.get(url)
            response.raise_for_status()  # Raise error if HTTP error
        
        # Parse RSS XML
        feed = feedparser.parse(response.content)
        
        # Process each entry in the feed
        for entry in feed.entries:
            try:
                # Extract article data
                article = _parse_entry(entry, source_name, category)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning("failed_to_parse_entry", 
                             source=source_name, error=str(e))
                continue
        
        logger.info("feed_fetched_successfully", 
                   source=source_name, 
                   articles_count=len(articles))
        
    except httpx.TimeoutException:
        logger.error("feed_timeout", source=source_name, url=url)
    except httpx.HTTPError as e:
        logger.error("feed_http_error", source=source_name, error=str(e))
    except Exception as e:
        logger.error("feed_fetch_failed", source=source_name, error=str(e))
    
    return articles


def _parse_entry(entry: Dict, source_name: str, category: str) -> Optional[Article]:
    """
    Parse a single RSS entry into an Article object.
    
    Args:
        entry: RSS entry dict from feedparser
        source_name: Name of the source
        category: Article category
    
    Returns:
        Article object or None if parsing fails
    """
    # Extract URL (required)
    url = entry.get('link')
    if not url:
        return None
    
    # Extract title (required)
    title = entry.get('title', 'No title')
    
    # Extract publication date
    published_at = _parse_date(entry)
    
    # Extract content snippet (try multiple fields)
    snippet = (
        entry.get('summary', '') or 
        entry.get('description', '') or 
        entry.get('content', [{}])[0].get('value', '')
    )
    
    # Clean up snippet (remove HTML tags, limit length)
    snippet = _clean_snippet(snippet)
    
    return Article(
        url=url,
        title=title,
        published_at=published_at,
        source=source_name,
        category=category,
        snippet=snippet
    )


def _parse_date(entry: Dict) -> datetime:
    """
    Parse publication date from RSS entry.
    Falls back to current time if parsing fails.
    """
    # Try multiple date fields
    date_str = (
        entry.get('published') or 
        entry.get('updated') or 
        entry.get('pubDate')
    )
    
    if date_str:
        try:
            # Parse date string to datetime
            dt = date_parser.parse(date_str)
            # Ensure timezone-aware (convert to UTC)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    
    # Fallback: current time
    return datetime.now(timezone.utc)


def _clean_snippet(text: str, max_length: int = 500) -> str:
    """
    Clean HTML tags and limit snippet length.
    """
    # Remove HTML tags (simple approach)
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length] + '...'
    
    return text


async def fetch_all_feeds(feed_configs: List[Dict]) -> List[Article]:
    """
    Fetch multiple RSS feeds concurrently.
    
    Args:
        feed_configs: List of feed config dicts with keys:
                      - url: Feed URL
                      - name: Source name
                      - category: Article category
    
    Returns:
        Combined list of all articles from all feeds
    """
    # Create tasks for each feed (runs concurrently)
    tasks = [
        fetch_feed(
            url=config['url'],
            source_name=config['name'],
            category=config['category']
        )
        for config in feed_configs
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine all articles
    all_articles = []
    for result in results:
        if isinstance(result, list):
            all_articles.extend(result)
        # Ignore exceptions (already logged)
    
    logger.info("all_feeds_fetched", 
               total_articles=len(all_articles),
               feeds_attempted=len(feed_configs))
    
    return all_articles
