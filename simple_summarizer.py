"""
LLM Summarization Module
Calls local Ollama to generate article summaries.
"""

import asyncio
from typing import List, Optional
import httpx
import json
import structlog

logger = structlog.get_logger()


class Summary:
    """Represents a summarized article"""
    
    def __init__(self, article_id: str, title: str, source: str, 
                 category: str, published_at, summary: str):
        self.article_id = article_id
        self.title = title
        self.source = source
        self.category = category
        self.published_at = published_at
        self.summary = summary
    
    def __repr__(self):
        return f"Summary({self.source}: {self.title[:40]}...)"


class OllamaClient:
    """
    Client for communicating with Ollama API.
    Handles model selection, timeouts, retries.
    """
    
    def __init__(self, host: str = "http://localhost:11434", 
                 model: str = "llama3.1:8b-instruct-q4_K_M",
                 timeout: int = 120):
        self.host = host
        self.model = model
        self.timeout = timeout
    
    async def generate(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama to generate text from a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Number of retry attempts if request fails
        
        Returns:
            Generated text or None if failed
        """
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.3,  # Lower = more focused/deterministic
                "top_p": 0.9,
                "num_predict": 200,  # Max tokens to generate
            }
        }
        
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.debug("calling_ollama", 
                               model=self.model, 
                               attempt=attempt + 1)
                    
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    generated_text = data.get("response", "").strip()
                    
                    if generated_text:
                        logger.debug("ollama_response_received", 
                                   length=len(generated_text))
                        return generated_text
                    else:
                        logger.warning("ollama_empty_response")
                        return None
            
            except httpx.TimeoutException:
                logger.error("ollama_timeout", 
                           model=self.model, 
                           attempt=attempt + 1)
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                return None
            
            except httpx.HTTPError as e:
                logger.error("ollama_http_error", 
                           error=str(e), 
                           attempt=attempt + 1)
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
                return None
            
            except Exception as e:
                logger.error("ollama_unexpected_error", 
                           error=str(e))
                return None
        
        return None


def build_summarization_prompt(title: str, snippet: str) -> str:
    """
    Construct a prompt for article summarization.
    
    Prompt engineering tips:
    - Be specific about output format
    - Give examples of good summaries (not shown here for brevity)
    - Set constraints (length, tone)
    """
    prompt = f"""You are a senior intelligence analyst writing for technical executives.

Summarize this news article in 2-3 sentences. Be factual, neutral, and concise.

ARTICLE TITLE: {title}
ARTICLE CONTENT: {snippet}

Guidelines:
- Lead with the core fact (what happened)
- Include key details (who, when, impact)
- Avoid speculation or editorializing
- Use active voice
- Maximum 80 words

Summary:"""
    
    return prompt


async def summarize_article(article, ollama_client: OllamaClient) -> Optional[Summary]:
    """
    Generate a summary for a single article.
    
    Args:
        article: Article object from ingestion
        ollama_client: OllamaClient instance
    
    Returns:
        Summary object or None if summarization fails
    """
    try:
        # Build prompt
        prompt = build_summarization_prompt(article.title, article.snippet)
        
        # Call LLM
        logger.info("summarizing_article", 
                   source=article.source, 
                   title=article.title[:50])
        
        summary_text = await ollama_client.generate(prompt)
        
        if not summary_text:
            logger.warning("summarization_failed_empty", 
                         article_id=article.id)
            return None
        
        # Create Summary object
        summary = Summary(
            article_id=article.id,
            title=article.title,
            source=article.source,
            category=article.category,
            published_at=article.published_at,
            summary=summary_text
        )
        
        logger.info("article_summarized", 
                   source=article.source,
                   summary_length=len(summary_text))
        
        return summary
    
    except Exception as e:
        logger.error("summarization_exception", 
                    article_id=article.id, 
                    error=str(e))
        return None


async def summarize_articles(articles: List, 
                            ollama_client: OllamaClient,
                            max_concurrent: int = 3) -> List[Summary]:
    """
    Summarize multiple articles with controlled concurrency.
    
    Args:
        articles: List of Article objects
        ollama_client: OllamaClient instance
        max_concurrent: Max number of simultaneous LLM calls
                       (GPU memory constraint)
    
    Returns:
        List of Summary objects
    """
    summaries = []
    
    # Process in batches to avoid overwhelming GPU
    for i in range(0, len(articles), max_concurrent):
        batch = articles[i:i + max_concurrent]
        
        logger.info("processing_batch", 
                   batch_number=i // max_concurrent + 1,
                   batch_size=len(batch))
        
        # Summarize batch concurrently
        tasks = [
            summarize_article(article, ollama_client)
            for article in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful summaries
        for result in results:
            if isinstance(result, Summary):
                summaries.append(result)
            # Skip exceptions (already logged)
        
        # Small delay between batches to let GPU cool down
        if i + max_concurrent < len(articles):
            await asyncio.sleep(1)
    
    logger.info("summarization_complete", 
               total_summaries=len(summaries),
               total_articles=len(articles))
    
    return summaries
