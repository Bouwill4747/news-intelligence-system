"""
Main Orchestrator for News Intelligence System
Coordinates ingestion, summarization, and output.
"""

import asyncio
from datetime import datetime
from pathlib import Path
import yaml
import structlog
from simple_ingestion import fetch_all_feeds
from simple_summarizer import OllamaClient, summarize_articles

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def load_sources_config(config_path: str = "config/sources.yaml") -> list:
    """
    Load RSS feed sources from YAML config.
    
    Returns:
        List of feed config dicts with keys: url, name, category, reliability
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        feeds = []
        
        # Flatten nested structure (category -> tier -> feeds)
        for category, tiers in config.items():
            for tier_name, feed_list in tiers.items():
                for feed in feed_list:
                    feeds.append({
                        'url': feed['url'],
                        'name': feed['name'],
                        'category': category,
                        'reliability': feed.get('reliability', 0.75)
                    })
        
        logger.info("sources_loaded", 
                   total_feeds=len(feeds),
                   categories=list(config.keys()))
        
        return feeds
    
    except FileNotFoundError:
        logger.error("config_not_found", path=config_path)
        return []
    except Exception as e:
        logger.error("config_load_error", error=str(e))
        return []


def format_digest(summaries: list) -> str:
    """
    Format summaries into a readable digest.
    
    Args:
        summaries: List of Summary objects
    
    Returns:
        Formatted string for display
    """
    # Group by category
    by_category = {}
    for summary in summaries:
        category = summary.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(summary)
    
    # Sort each category by publication date (newest first)
    for category in by_category:
        by_category[category].sort(
            key=lambda s: s.published_at, 
            reverse=True
        )
    
    # Build output
    output = []
    output.append("=" * 80)
    output.append(f"🌐 DAILY NEWS INTELLIGENCE DIGEST")
    output.append(f"📅 {datetime.now().strftime('%B %d, %Y - %H:%M UTC')}")
    output.append(f"📊 {len(summaries)} articles summarized")
    output.append("=" * 80)
    
    # Category emoji mapping
    category_emoji = {
        'cybersecurity': '🔐',
        'technology': '💻',
        'crypto': '₿'
    }
    
    for category, items in by_category.items():
        emoji = category_emoji.get(category, '📰')
        output.append(f"\n\n{emoji} {category.upper()}")
        output.append("-" * 80)
        
        for summary in items:
            output.append(f"\n📰 {summary.title}")
            output.append(f"   Source: {summary.source}")
            output.append(f"   Published: {summary.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
            output.append(f"\n   {summary.summary}")
            output.append("")
    
    output.append("\n" + "=" * 80)
    output.append("End of digest")
    output.append("=" * 80)
    
    return "\n".join(output)


async def run_digest(limit_articles: int = None):
    """
    Main pipeline execution.
    
    Args:
        limit_articles: Optional limit for testing (e.g., 10 articles)
    """
    start_time = datetime.now()
    logger.info("digest_started")
    
    try:
        # Step 1: Load configuration
        logger.info("step_1_loading_config")
        feeds = load_sources_config()
        
        if not feeds:
            logger.error("no_feeds_configured")
            return
        
        logger.info("feeds_loaded", count=len(feeds))
        
        # Step 2: Fetch articles
        logger.info("step_2_fetching_articles")
        articles = await fetch_all_feeds(feeds)
        
        if not articles:
            logger.warning("no_articles_fetched")
            return
        
        # Optional: Limit for testing
        if limit_articles:
            articles = articles[:limit_articles]
            logger.info("articles_limited_for_testing", count=limit_articles)
        
        logger.info("articles_fetched", count=len(articles))
        
        # Step 3: Summarize with LLM
        logger.info("step_3_summarizing_articles")
        ollama = OllamaClient(
            host="http://localhost:11434",
            model="llama3.1:8b-instruct-q4_K_M",
            timeout=120
        )
        
        summaries = await summarize_articles(
            articles, 
            ollama, 
            max_concurrent=3  # GPU-friendly batching
        )
        
        if not summaries:
            logger.warning("no_summaries_generated")
            return
        
        logger.info("summaries_generated", count=len(summaries))
        
        # Step 4: Format output
        logger.info("step_4_formatting_digest")
        digest = format_digest(summaries)
        
        # Step 5: Display
        print("\n" + digest)
        
        # Stats
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("digest_completed",
                   duration_seconds=round(elapsed, 2),
                   articles_processed=len(articles),
                   summaries_created=len(summaries))
        
    except Exception as e:
        logger.error("digest_failed", error=str(e), exc_info=True)
        raise


def main():
    """Entry point"""
    import sys
    
    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode: limit to 10 articles for quick verification
        print("🧪 Running in TEST mode (10 articles)")
        asyncio.run(run_digest(limit_articles=10))
    else:
        # Full mode: process all articles
        print("🚀 Running FULL digest (all articles)")
        asyncio.run(run_digest())


if __name__ == "__main__":
    main()
