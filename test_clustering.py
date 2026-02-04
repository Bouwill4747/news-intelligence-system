import asyncio
from simple_ingestion import fetch_all_feeds
from advanced_clustering import cluster_articles_async
import structlog
import yaml

# Setup logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)

async def main():
    print("Step 1: Loading sources...")
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
    
    print(f"✅ Loaded {len(feeds)} feeds\n")
    
    print("Step 2: Fetching articles...")
    articles = await fetch_all_feeds(feeds)
    print(f"✅ Fetched {len(articles)} articles\n")
    
    print("Step 3: Clustering (this will take ~30 seconds - downloading model + computing embeddings)...")
    clusters = await cluster_articles_async(articles)
    print(f"✅ Created {len(clusters)} clusters\n")
    
    # Show results by category
    by_category = {}
    for cluster in clusters:
        cat = cluster.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(cluster)
    
    print("="*70)
    print("CLUSTERING RESULTS")
    print("="*70)
    
    for category, cat_clusters in by_category.items():
        print(f"\n{category.upper()}: {len(cat_clusters)} clusters")
        
        # Show first 5 clusters
        for i, cluster in enumerate(cat_clusters[:5]):
            print(f"\n  Cluster {i+1}: {cluster.article_count} articles")
            print(f"  Topic: {cluster.topic_label[:70]}...")
            
            # Show article titles in this cluster
            for article in cluster.articles[:3]:  # Show max 3
                print(f"    - {article.source}: {article.title[:60]}...")
            
            if cluster.article_count > 3:
                print(f"    ... and {cluster.article_count - 3} more")
        
        if len(cat_clusters) > 5:
            print(f"\n  ... and {len(cat_clusters) - 5} more clusters")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {len(articles)} articles → {len(clusters)} clusters")
    print(f"Reduction: {len(articles) - len(clusters)} duplicate articles removed")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
