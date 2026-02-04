import asyncio
from simple_ingestion import fetch_all_feeds
import structlog

# Setup logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)

# Test with 3 feeds
feeds = [
    {
        'url': 'https://krebsonsecurity.com/feed/',
        'name': 'Krebs on Security',
        'category': 'cybersecurity'
    },
    {
        'url': 'https://www.bleepingcomputer.com/feed/',
        'name': 'BleepingComputer',
        'category': 'cybersecurity'
    },
    {
        'url': 'https://techcrunch.com/feed/',
        'name': 'TechCrunch',
        'category': 'technology'
    },
]

async def main():
    print("Fetching feeds...")
    articles = await fetch_all_feeds(feeds)
    
    print(f"\n✅ Fetched {len(articles)} articles")
    print("\nFirst 3 articles:")
    for article in articles[:3]:
        print(f"\n  📰 {article.title}")
        print(f"     Source: {article.source}")
        print(f"     Published: {article.published_at}")
        print(f"     Snippet: {article.snippet[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
