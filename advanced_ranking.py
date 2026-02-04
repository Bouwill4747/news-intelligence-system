"""
Advanced Ranking Module
Scores clusters by importance using multiple signals + LLM judgment.
Early-exit strategy: stop calling LLM once we have enough high-scoring clusters.
"""

import asyncio
from typing import List, Dict
from datetime import datetime, timezone
import json
import structlog
from simple_summarizer import OllamaClient

logger = structlog.get_logger()


class ClusterScore:
    """Represents a scored cluster"""

    def __init__(self, cluster, score: float, reasoning: str = ""):
        self.cluster = cluster
        self.score = score
        self.reasoning = reasoning

    def __repr__(self):
        return f"ClusterScore(score={self.score:.1f}, topic='{self.cluster.topic_label[:40]}...')"


class ClusterRanker:

    def __init__(self, ollama_client: OllamaClient, recency_decay_hours: int = 24):
        self.ollama_client = ollama_client
        self.recency_decay_hours = recency_decay_hours

        self.source_scores = {
            'Krebs on Security': 0.95,
            'BleepingComputer': 0.90,
            'The Hacker News': 0.85,
            'Dark Reading': 0.80,
            'SecurityWeek': 0.75,
            'CISA Alerts': 1.0,
            'Ars Technica': 0.95,
            'The Verge': 0.85,
            'TechCrunch': 0.80,
            'Wired': 0.85,
            'MIT Technology Review': 0.90,
            'VentureBeat': 0.75,
            'Hacker News': 0.70,
            'CoinDesk': 0.85,
            'The Block': 0.85,
            'Decrypt': 0.80,
            'Cointelegraph': 0.70,
            'Bitcoin Magazine': 0.75
        }

    def calculate_source_score(self, cluster) -> float:
        scores = [self.source_scores.get(a.source, 0.70) for a in cluster.articles]
        return (sum(scores) / len(scores)) * 100

    def calculate_recency_score(self, cluster) -> float:
        now = datetime.now(timezone.utc)
        most_recent = max(cluster.articles, key=lambda a: a.published_at)
        hours_old = max(0, (now - most_recent.published_at).total_seconds() / 3600)
        return 100 * (0.5 ** (hours_old / self.recency_decay_hours))

    def calculate_cluster_size_score(self, cluster) -> float:
        n = cluster.article_count
        if n == 1: return 20
        if n == 2: return 40
        if n == 3: return 55
        if n == 4: return 70
        return min(100, 70 + (n - 5) * 6)

    def calculate_rule_based_score(self, cluster) -> float:
        """
        Full score using ONLY rules — no LLM.
        Used for sorting priority and for clusters we skip LLM on.
        Assumes LLM would give 60 (neutral middle ground).
        """
        source = self.calculate_source_score(cluster)
        recency = self.calculate_recency_score(cluster)
        size = self.calculate_cluster_size_score(cluster)
        assumed_llm = 60.0  # neutral assumption
        return source * 0.20 + recency * 0.20 + size * 0.15 + assumed_llm * 0.45

    async def calculate_llm_score(self, cluster) -> tuple:
        prompt = self._build_ranking_prompt(cluster)
        response = await self.ollama_client.generate(prompt)

        if not response:
            return 50.0, "LLM response empty"

        try:
            response = response.strip()
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1])

            data = json.loads(response)
            score = max(0, min(100, float(data.get('score', 50))))
            reasoning = data.get('reasoning', 'No reasoning provided')

            logger.info("llm_score_received",
                       cluster_id=cluster.id, score=score,
                       reasoning=reasoning[:100])
            return score, reasoning

        except (json.JSONDecodeError, Exception) as e:
            logger.error("llm_score_parse_error",
                        cluster_id=cluster.id, error=str(e))
            return 50.0, "Failed to parse LLM response"

    def _build_ranking_prompt(self, cluster) -> str:
        articles_text = "\n".join([
            f"- {a.source}: {a.title}" for a in cluster.articles[:5]
        ])
        if cluster.article_count > 5:
            articles_text += f"\n... and {cluster.article_count - 5} more articles"

        return f"""You are a news editor for a global intelligence brief. Your audience is security researchers, engineers, and tech executives.

Evaluate this news cluster for importance on a scale of 0-100:

TOPIC: {cluster.topic_label}
CATEGORY: {cluster.category}
NUMBER OF SOURCES: {cluster.article_count}
ARTICLES:
{articles_text}

Scoring criteria:
- Global impact (affects many users/organizations)
- Technical significance (breakthrough, vulnerability, major outage)
- Policy/regulatory implications
- Market-moving potential

Score 80-100: Critical (major breach, zero-day, regulation change, market crash)
Score 60-79: High (significant product launch, merger, vulnerability, major policy)
Score 40-59: Medium (interesting research, minor outage, acquisition, price movement)
Score 0-39: Low (incremental updates, rumors, clickbait, minor features)

Respond ONLY with a JSON object (no markdown, no extra text):
{{"score": <number 0-100>, "reasoning": "<2-sentence explanation>"}}"""

    async def score_cluster_with_llm(self, cluster) -> ClusterScore:
        """Score a single cluster using LLM for the final component."""
        source = self.calculate_source_score(cluster)
        recency = self.calculate_recency_score(cluster)
        size = self.calculate_cluster_size_score(cluster)
        llm_score, reasoning = await self.calculate_llm_score(cluster)

        final = source * 0.20 + recency * 0.20 + size * 0.15 + llm_score * 0.45

        logger.info("cluster_scored",
                   cluster_id=cluster.id,
                   topic=cluster.topic_label[:50],
                   final_score=round(final, 1),
                   source=round(source, 1),
                   recency=round(recency, 1),
                   size=round(size, 1),
                   llm=round(llm_score, 1))

        return ClusterScore(cluster=cluster, score=final, reasoning=reasoning)

    async def rank_clusters(self, clusters: List, top_n: int = 15, min_score: float = 60.0) -> List[ClusterScore]:
        """
        Smart ranking strategy:
        1. Score ALL clusters with rules only (instant, no GPU)
        2. Sort by rule-based score (best candidates first)
        3. Call LLM only on the top candidates until we have enough confirmed stories
        4. Stop early once we have top_n clusters above min_score

        This means instead of 75 LLM calls we might only need 20-25.
        """
        logger.info("ranking_clusters", total=len(clusters))

        # Step 1: Rule-based pre-score everything (instant)
        rule_scored = []
        for cluster in clusters:
            rule_score = self.calculate_rule_based_score(cluster)
            rule_scored.append((cluster, rule_score))

        # Step 2: Sort by rule score — best candidates first
        rule_scored.sort(key=lambda x: x[1], reverse=True)

        logger.info("rule_based_sort_done",
                   top_3_scores=[round(s, 1) for _, s in rule_scored[:3]],
                   bottom_3_scores=[round(s, 1) for _, s in rule_scored[-3:]])

        # Step 3: Call LLM on candidates in order, stop when we have enough
        confirmed = []       # Clusters that passed with LLM confirmation
        llm_calls = 0

        for cluster, rule_score in rule_scored:
            # If even a perfect LLM score (100) can't reach min_score, stop entirely
            source = self.calculate_source_score(cluster)
            recency = self.calculate_recency_score(cluster)
            size = self.calculate_cluster_size_score(cluster)
            max_possible = source * 0.20 + recency * 0.20 + size * 0.15 + 100 * 0.45

            if max_possible < min_score:
                logger.info("early_exit_max_possible",
                           remaining_clusters=len(rule_scored) - llm_calls,
                           max_possible=round(max_possible, 1))
                break

            # Call LLM
            scored = await self.score_cluster_with_llm(cluster)
            llm_calls += 1

            if scored.score >= min_score:
                confirmed.append(scored)

            # EARLY EXIT: we have enough confirmed stories
            if len(confirmed) >= top_n:
                logger.info("early_exit_enough_stories",
                           confirmed=len(confirmed),
                           llm_calls=llm_calls,
                           clusters_skipped=len(clusters) - llm_calls)
                break

        # If we didn't get enough confirmed, add best unconfirmed ones
        # (rule-based score only, no LLM) to fill the digest
        if len(confirmed) < top_n:
            logger.info("filling_remaining_with_rule_based",
                       confirmed=len(confirmed),
                       need=top_n - len(confirmed))
            # Gather clusters we didn't LLM-score
            llm_scored_ids = {cs.cluster.id for cs in confirmed}
            for cluster, rule_score in rule_scored:
                if cluster.id not in llm_scored_ids and rule_score >= min_score * 0.85:
                    confirmed.append(ClusterScore(
                        cluster=cluster,
                        score=rule_score,
                        reasoning="Ranked by rules (source + recency + coverage)"
                    ))
                    if len(confirmed) >= top_n:
                        break

        confirmed.sort(key=lambda x: x.score, reverse=True)

        logger.info("ranking_complete",
                   total_clusters=len(clusters),
                   llm_calls_made=llm_calls,
                   confirmed_stories=len(confirmed))

        return confirmed[:top_n]


async def rank_and_filter_clusters(clusters: List,
                                   top_n: int = 15,
                                   min_score: float = 60.0) -> List[ClusterScore]:
    """
    High-level function: pre-filter, then rank with early exit.
    """
    # Pre-filter: drop singletons (single article, single source)
    candidates = []
    skipped = 0
    for c in clusters:
        if c.article_count == 1 and len(set(a.source for a in c.articles)) == 1:
            skipped += 1
            continue
        candidates.append(c)

    logger.info("pre_filter_done",
               original=len(clusters),
               candidates=len(candidates),
               skipped=skipped)

    ollama_client = OllamaClient(
        host="http://localhost:11434",
        model="llama3.2:3b-instruct-q4_K_M",
        timeout=60
    )

    ranker = ClusterRanker(
        ollama_client=ollama_client,
        recency_decay_hours=24
    )

    return await ranker.rank_clusters(candidates, top_n=top_n, min_score=min_score)
