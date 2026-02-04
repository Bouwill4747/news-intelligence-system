import asyncio
from simple_ingestion import fetch_all_feeds
from advanced_clustering import cluster_articles_async
from advanced_ranking import rank_and_filter_clusters
import structlog
import yaml

# Setup logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)

async def main():
    print("Step 1: Fetching articles...")
    
    with open('config/sources.yaml') as f:
        config = yaml.safe_load(f)
    
    feeds = []
    for category, tiers in config.items():
        for tier_name, feed_list in tiers.items():
            for feed in feed_list:
                feeds.append({
                    'url': feed['url'],
                    'name': feed['name'],
                    'category': category
                })
    
    articles = await fetch_all_feeds(feeds)
    print(f"✅ Fetched {len(articles)} articles\n")
    
    print("Step 2: Clustering...")
    clusters = await cluster_articles_async(articles)
    print(f"✅ Created {len(clusters)} clusters\n")
    
    print("Step 3: Ranking clusters (this will take ~2 minutes - LLM scoring each cluster)...")
    print("(Using Llama 3.2 3B for fast classification)\n")
    
    top_clusters = await rank_and_filter_clusters(
        clusters,
        top_n=15,
        min_score=60.0
    )
    
    print(f"\n✅ Selected {len(top_clusters)} top stories\n")
    
    print("="*80)
    print("TOP 15 STORIES (Ranked by Importance)")
    print("="*80)
    
    for i, scored in enumerate(top_clusters, 1):
        cluster = scored.cluster
        print(f"\n#{i} | Score: {scored.score:.1f}/100 | Category: {cluster.category.upper()}")
        print(f"Topic: {cluster.topic_label}")
        print(f"Sources: {cluster.article_count} articles")
        print(f"Reasoning: {scored.reasoning}")
        print("-"*80)
    
    print("\n" + "="*80)
    print(f"SUMMARY: {len(articles)} articles → {len(clusters)} clusters → {len(top_clusters)} top stories")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
