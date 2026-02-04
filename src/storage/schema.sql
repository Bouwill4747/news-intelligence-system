-- News Intelligence System Database Schema

-- Raw articles from RSS feeds
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    snippet TEXT,
    embedding BLOB,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_processed ON articles(processed);

-- Clusters (deduplicated topics)
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    representative_article_id TEXT,
    topic_label TEXT,
    importance_score REAL,
    summary TEXT,
    article_count INTEGER,
    category TEXT,
    FOREIGN KEY (representative_article_id) REFERENCES articles(id)
);

CREATE INDEX IF NOT EXISTS idx_clusters_score ON clusters(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_clusters_category ON clusters(category);

-- Many-to-many: clusters <-> articles
CREATE TABLE IF NOT EXISTS cluster_articles (
    cluster_id INTEGER,
    article_id TEXT,
    similarity_score REAL,
    PRIMARY KEY (cluster_id, article_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id),
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

-- Delivery log
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_date DATE UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    telegram_sent BOOLEAN DEFAULT FALSE,
    email_sent BOOLEAN DEFAULT FALSE,
    cluster_ids TEXT,  -- JSON array
    article_count INTEGER,
    cluster_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_digests_date ON digests(digest_date DESC);
