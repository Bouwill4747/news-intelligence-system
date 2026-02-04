import asyncio
from simple_ingestion import fetch_all_feeds
from simple_summarizer import OllamaClient, summarize_articles
import structlog

# Setup logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)

# Test with 1 feed, limit to 3 articles
feeds = [{
    'url': 'https://krebsonsecurity.com/feed/',
    'name': 'Krebs on Security',
    'category': 'cybersecurity'
}]

async def main():
    print("Step 1: Fetching articles...")
    articles = await fetch_all_feeds(feeds)
    
    # Take only first 3 articles for quick test
    articles = articles[:3]
    print(f"✅ Got {len(articles)} articles\n")
    
    print("Step 2: Summarizing with Ollama...")
    print("(This will take ~6 seconds on your GPU)\n")
    
    ollama = OllamaClient(
        host="http://localhost:11434",
        model="llama3.1:8b-instruct-q4_K_M"
    )
    
    summaries = await summarize_articles(articles, ollama, max_concurrent=3)
    
    print(f"\n✅ Generated {len(summaries)} summaries\n")
    print("="*70)
    
    for summary in summaries:
        print(f"\n📰 {summary.title}")
        print(f"   Source: {summary.source}")
        print(f"   Category: {summary.category}")
        print(f"\n   Summary:")
        print(f"   {summary.summary}")
        print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(main())
