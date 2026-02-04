"""
Advanced Main Orchestrator
Full pipeline: Ingest → Cluster → Rank → Summarize → Format → Output
"""

import asyncio
from datetime import datetime
from pathlib import Path
import yaml
import structlog

from simple_ingestion import fetch_all_feeds
from simple_summarizer import OllamaClient, summarize_article
from advanced_clustering import cluster_articles_async
from advanced_ranking import rank_and_filter_clusters

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


class DigestSummary:
    """Represents a final summarized cluster"""
    
    def __init__(self, cluster_score, summary_text: str):
        self.cluster = cluster_score.cluster
        self.score = cluster_score.score
        self.reasoning = cluster_score.reasoning
        self.summary = summary_text
        self.category = self.cluster.category
        self.topic = self.cluster.topic_label
        self.article_count = self.cluster.article_count
        self.sources = list(set([a.source for a in self.cluster.articles]))
        self.published_at = self.cluster.representative.published_at


def load_sources_config(config_path: str = "config/sources.yaml") -> list:
    """Load RSS feed sources from YAML config"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        feeds = []
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


def build_cluster_summarization_prompt(cluster) -> str:
    """
    Build a prompt to summarize an entire cluster (not individual articles).
    
    This is DIFFERENT from simple summarization:
    - Input: Multiple articles about the same topic
    - Output: One unified summary covering all angles
    """
    # Get all article titles and snippets
    articles_text = ""
    for i, article in enumerate(cluster.articles[:5], 1):  # Max 5 for prompt length
        articles_text += f"\n{i}. {article.source}: {article.title}\n"
        articles_text += f"   {article.snippet[:200]}...\n"
    
    if cluster.article_count > 5:
        articles_text += f"\n... and {cluster.article_count - 5} more articles\n"
    
    prompt = f"""You are a senior intelligence analyst writing for technical executives.

Summarize this news topic in 3-4 sentences. Synthesize information from ALL articles below into ONE coherent summary.

TOPIC: {cluster.topic_label}
CATEGORY: {cluster.category}
SOURCES: {cluster.article_count} articles from {len(set([a.source for a in cluster.articles]))} different sources

ARTICLES:
{articles_text}

Guidelines:
- Lead with the core fact (what happened)
- Synthesize key details from multiple sources
- Include technical specifics (numbers, names, impact)
- Avoid speculation or editorializing
- Use active voice
- Maximum 100 words

Summary:"""
    
    return prompt


async def summarize_cluster(cluster_score, ollama_client: OllamaClient) -> DigestSummary:
    """
    Generate a summary for a scored cluster.
    
    Args:
        cluster_score: ClusterScore object
        ollama_client: OllamaClient instance
    
    Returns:
        DigestSummary object
    """
    cluster = cluster_score.cluster
    
    try:
        # Build prompt for cluster summarization
        prompt = build_cluster_summarization_prompt(cluster)
        
        logger.info("summarizing_cluster",
                   cluster_id=cluster.id,
                   topic=cluster.topic_label[:50],
                   articles=cluster.article_count)
        
        # Call LLM (using Llama 3.1 8B for quality)
        summary_text = await ollama_client.generate(prompt)
        
        if not summary_text:
            logger.warning("summarization_failed",
                         cluster_id=cluster.id)
            # Fallback: use representative article snippet
            summary_text = cluster.representative.snippet[:300]
        
        # Clean up summary
        summary_text = summary_text.strip()
        if summary_text.lower().startswith("here is"):
            # Remove "Here is a summary:" preamble
            lines = summary_text.split('\n')
            summary_text = '\n'.join(lines[1:]).strip()
        
        logger.info("cluster_summarized",
                   cluster_id=cluster.id,
                   summary_length=len(summary_text))
        
        return DigestSummary(
            cluster_score=cluster_score,
            summary_text=summary_text
        )
    
    except Exception as e:
        logger.error("summarization_exception",
                    cluster_id=cluster.id,
                    error=str(e))
        # Fallback summary
        return DigestSummary(
            cluster_score=cluster_score,
            summary_text=cluster.representative.snippet[:300]
        )


async def summarize_top_clusters(scored_clusters: list,
                                 max_concurrent: int = 3) -> list:
    """
    Summarize all top-ranked clusters.
    
    Args:
        scored_clusters: List of ClusterScore objects
        max_concurrent: Max simultaneous LLM calls
    
    Returns:
        List of DigestSummary objects
    """
    logger.info("summarizing_top_clusters", count=len(scored_clusters))
    
    # Use Llama 3.1 8B for high-quality summaries
    ollama_client = OllamaClient(
        host="http://localhost:11434",
        model="llama3.1:8b-instruct-q4_K_M",
        timeout=120
    )
    
    summaries = []
    
    # Process in batches
    for i in range(0, len(scored_clusters), max_concurrent):
        batch = scored_clusters[i:i + max_concurrent]
        
        logger.info("summarizing_batch",
                   batch_num=i // max_concurrent + 1,
                   batch_size=len(batch))
        
        # Summarize batch concurrently
        tasks = [
            summarize_cluster(cluster_score, ollama_client)
            for cluster_score in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful summaries
        for result in results:
            if isinstance(result, DigestSummary):
                summaries.append(result)
        
        # Small delay between batches
        if i + max_concurrent < len(scored_clusters):
            await asyncio.sleep(1)
    
    logger.info("summarization_complete", total=len(summaries))
    
    return summaries


def format_advanced_digest(summaries: list) -> str:
    """
    Format summaries into final digest output.
    
    Args:
        summaries: List of DigestSummary objects (already sorted by score)
    
    Returns:
        Formatted string
    """
    # Group by category
    by_category = {}
    for summary in summaries:
        category = summary.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(summary)
    
    # Build output
    output = []
    output.append("=" * 80)
    output.append("🌐 DAILY INTELLIGENCE DIGEST (AI-POWERED)")
    output.append(f"📅 {datetime.now().strftime('%B %d, %Y - %H:%M UTC')}")
    output.append(f"📊 {len(summaries)} high-priority stories (from {sum(s.article_count for s in summaries)} articles)")
    output.append("=" * 80)
    
    # Category emoji mapping
    category_emoji = {
        'cybersecurity': '🔐',
        'technology': '💻',
        'crypto': '₿'
    }
    
    # Output by category
    for category in ['cybersecurity', 'technology', 'crypto']:
        if category not in by_category:
            continue
        
        items = by_category[category]
        emoji = category_emoji.get(category, '📰')
        
        output.append(f"\n\n{emoji} {category.upper()}")
        output.append("-" * 80)
        
        for i, summary in enumerate(items, 1):
            output.append(f"\n📰 Story #{items.index(summary) + 1}")
            output.append(f"   Topic: {summary.topic}")
            output.append(f"   Importance: {summary.score:.1f}/100")
            output.append(f"   Sources: {summary.article_count} articles ({', '.join(summary.sources[:3])})")
            output.append(f"   Published: {summary.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
            output.append(f"\n   Summary:")
            
            # Wrap summary text nicely
            import textwrap
            wrapped = textwrap.fill(summary.summary, width=75, initial_indent='   ', subsequent_indent='   ')
            output.append(wrapped)
            
            output.append(f"\n   Why important: {summary.reasoning}")
            output.append("")
    
    output.append("\n" + "=" * 80)
    output.append("End of digest")
    output.append("=" * 80)
    
    return "\n".join(output)


async def run_advanced_digest(top_n: int = 15, min_score: float = 60.0):
    """
    Main pipeline: Full intelligence digest generation.
    
    Args:
        top_n: Number of top stories to include
        min_score: Minimum importance score threshold
    """
    start_time = datetime.now()
    logger.info("advanced_digest_started")
    
    try:
        # Step 1: Load configuration
        logger.info("step_1_loading_config")
        feeds = load_sources_config()
        
        if not feeds:
            logger.error("no_feeds_configured")
            return
        
        # Step 2: Fetch articles
        logger.info("step_2_fetching_articles")
        articles = await fetch_all_feeds(feeds)
        
        if not articles:
            logger.warning("no_articles_fetched")
            return
        
        logger.info("articles_fetched", count=len(articles))
        
        # Step 3: Cluster articles (deduplicate)
        logger.info("step_3_clustering_articles")
        clusters = await cluster_articles_async(articles)
        
        logger.info("clusters_created", count=len(clusters))
        
        # Step 4: Rank and filter clusters
        logger.info("step_4_ranking_clusters")
        top_clusters = await rank_and_filter_clusters(
            clusters,
            top_n=top_n,
            min_score=min_score
        )
        
        if not top_clusters:
            logger.warning("no_clusters_passed_threshold")
            return
        
        logger.info("top_clusters_selected", count=len(top_clusters))
        
        # Step 5: Summarize top clusters
        logger.info("step_5_summarizing_clusters")
        summaries = await summarize_top_clusters(top_clusters, max_concurrent=3)
        
        logger.info("summaries_generated", count=len(summaries))
        
        # Step 6: Format digest
        logger.info("step_6_formatting_digest")
        digest = format_advanced_digest(summaries)
        
        # Step 7: Output
        print("\n" + digest)
        
	# Step 9: Deliver via Telegram and Email
        logger.info("step_9_delivering_digest")
        from delivery import deliver_digest
        delivery_results = await deliver_digest(digest)
        
        if delivery_results['telegram']:
            print("✅ Telegram delivery successful!")
        else:
            print("❌ Telegram delivery failed!")
        
        if delivery_results['email']:
            print("✅ Email delivery successful!")
        else:
            print("❌ Email delivery failed!")

        # Stats
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("advanced_digest_completed",
                   duration_seconds=round(elapsed, 2),
                   articles_fetched=len(articles),
                   clusters_created=len(clusters),
                   top_stories=len(summaries))
        
    except Exception as e:
        logger.error("digest_failed", error=str(e), exc_info=True)
        raise


def main():
    """Entry point"""
    print("🚀 Running ADVANCED Intelligence Digest")
    print("   Pipeline: Fetch → Cluster → Rank → Summarize → Output\n")
    
    asyncio.run(run_advanced_digest(top_n=15, min_score=60.0))


if __name__ == "__main__":
    main()
