"""
Advanced Clustering Module
Uses embeddings to deduplicate and group similar articles into topics.
"""

import asyncio
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import structlog

logger = structlog.get_logger()


class Cluster:
    """Represents a cluster of related articles"""
    
    def __init__(self, cluster_id: int, articles: List, category: str):
        self.id = cluster_id
        self.articles = articles
        self.category = category
        self.article_count = len(articles)
        
        # Select representative article (most recent)
        self.representative = max(articles, key=lambda a: a.published_at)
        
        # Generate topic label from titles
        self.topic_label = self._generate_topic_label()
    
    def _generate_topic_label(self) -> str:
        """
        Generate a topic label from article titles.
        For now, just use the representative article's title.
        In production, you'd use an LLM to generate a better label.
        """
        return self.representative.title
    
    def __repr__(self):
        return f"Cluster(id={self.id}, articles={self.article_count}, topic='{self.topic_label[:50]}...')"


class EmbeddingEngine:
    """
    Handles text-to-vector conversion using sentence transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name
                       'all-MiniLM-L6-v2' = 80MB, fast, 384 dimensions
        """
        logger.info("loading_embedding_model", model=model_name)
        
        # Load model (downloads on first run, ~80MB)
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info("embedding_model_loaded", 
                   model=model_name,
                   dimensions=self.embedding_dim)
    
    def embed_articles(self, articles: List) -> np.ndarray:
        """
        Convert articles to embeddings.
        
        Args:
            articles: List of Article objects
        
        Returns:
            numpy array of shape (n_articles, embedding_dim)
        """
        # Combine title + snippet for better semantic representation
        texts = [
            f"{article.title} {article.snippet}"
            for article in articles
        ]
        
        logger.info("generating_embeddings", count=len(texts))
        
        # Generate embeddings (batch processing for speed)
        # This runs on CPU, takes ~50ms per article
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Process 32 at a time
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.info("embeddings_generated", 
                   shape=embeddings.shape,
                   dtype=embeddings.dtype)
        
        return embeddings


class ArticleClusterer:
    """
    Groups similar articles into clusters using HDBSCAN.
    """
    
    def __init__(self, 
                 min_cluster_size: int = 2,
                 min_samples: int = 1,
                 similarity_threshold: float = 0.75):
        """
        Initialize clusterer.
        
        Args:
            min_cluster_size: Minimum articles to form a cluster
            min_samples: HDBSCAN parameter (1 = more clusters)
            similarity_threshold: Cosine similarity cutoff (0.75 = fairly similar)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
    
    def cluster_articles(self, 
                        articles: List, 
                        embeddings: np.ndarray) -> List[Cluster]:
        """
        Cluster articles based on embedding similarity.
        
        Args:
            articles: List of Article objects
            embeddings: numpy array of embeddings
        
        Returns:
            List of Cluster objects
        """
        if len(articles) < self.min_cluster_size:
            logger.warning("too_few_articles_to_cluster", count=len(articles))
            # Return each article as its own cluster
            return [
                Cluster(cluster_id=i, articles=[article], category=article.category)
                for i, article in enumerate(articles)
            ]
        
        logger.info("clustering_articles", 
                   count=len(articles),
                   embedding_shape=embeddings.shape)
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',  # Works well with normalized embeddings
            cluster_selection_method='eom'  # Excess of Mass
        )
        
        # Fit and predict cluster labels
        # Returns array like: [0, 0, 1, -1, 2, 2, 1, -1, ...]
        # Where -1 = noise (doesn't belong to any cluster)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Group articles by cluster label
        clusters_dict = {}
        noise_articles = []
        
        for article, label in zip(articles, cluster_labels):
            if label == -1:
                # Noise: treat as singleton cluster
                noise_articles.append(article)
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(article)
        
        # Create Cluster objects
        clusters = []
        
        # Regular clusters
        for cluster_id, cluster_articles in clusters_dict.items():
            # All articles in a cluster should have same category
            # (we cluster per-category, so this is guaranteed)
            category = cluster_articles[0].category
            
            cluster = Cluster(
                cluster_id=len(clusters),
                articles=cluster_articles,
                category=category
            )
            clusters.append(cluster)
        
        # Noise articles become singleton clusters
        for article in noise_articles:
            cluster = Cluster(
                cluster_id=len(clusters),
                articles=[article],
                category=article.category
            )
            clusters.append(cluster)
        
        logger.info("clustering_complete",
                   total_clusters=len(clusters),
                   regular_clusters=len(clusters_dict),
                   singleton_clusters=len(noise_articles))
        
        return clusters


def cluster_by_category(articles: List, 
                       embedding_engine: EmbeddingEngine,
                       clusterer: ArticleClusterer) -> List[Cluster]:
    """
    Cluster articles separately by category, then combine.
    
    Why cluster by category?
    - Prevents cross-category confusion (crypto + cybersecurity)
    - Better clustering quality within each domain
    
    Args:
        articles: List of all articles
        embedding_engine: EmbeddingEngine instance
        clusterer: ArticleClusterer instance
    
    Returns:
        Combined list of Cluster objects
    """
    # Group articles by category
    by_category = {}
    for article in articles:
        category = article.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(article)
    
    logger.info("clustering_by_category", 
               categories=list(by_category.keys()),
               counts={k: len(v) for k, v in by_category.items()})
    
    all_clusters = []
    
    # Cluster each category separately
    for category, category_articles in by_category.items():
        logger.info("clustering_category", 
                   category=category,
                   articles=len(category_articles))
        
        # Generate embeddings for this category
        embeddings = embedding_engine.embed_articles(category_articles)
        
        # Cluster
        clusters = clusterer.cluster_articles(category_articles, embeddings)
        
        # Add to results
        all_clusters.extend(clusters)
        
        logger.info("category_clustered",
                   category=category,
                   clusters=len(clusters))
    
    logger.info("all_categories_clustered", total_clusters=len(all_clusters))
    
    return all_clusters


# High-level function for easy use
async def cluster_articles_async(articles: List) -> List[Cluster]:
    """
    Main entry point: cluster articles into topics.
    
    Args:
        articles: List of Article objects
    
    Returns:
        List of Cluster objects
    """
    # Initialize engines
    embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    clusterer = ArticleClusterer(
        min_cluster_size=2,
        min_samples=1,
        similarity_threshold=0.75
    )
    
    # Cluster by category
    clusters = cluster_by_category(articles, embedding_engine, clusterer)
    
    return clusters
