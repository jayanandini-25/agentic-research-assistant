"""
app/retrieval/hackernews_retriever.py  (FIXED v4)

Fixes vs v3
-----------
FIX 1 — C++ Header Units talk ([76]) appearing for "challenges implementing RAG".

  Root cause A: `_extract_hn_query()` fell through all tech-term patterns
  and hit the fallback:
      words = ["challenges", "implementing", "retrieval", "augmented"]
      → " ".join(words[:3]) = "challenges implementing retrieval"
  But in the second pass (len(results) < 2), it trimmed FURTHER:
      fallback_query = "challenges implementing"  (only 2 words)
  "challenges implementing" returned "The Challenges of Implementing C++ Header Units"
  which scored 0.21 on relevance_score() because "challenges" and "implementing"
  both match tokens in "What are the key challenges in implementing RAG".

  Root cause B: _MIN_RELEVANCE = 0.15 was too low. Anything with 2 shared
  stopword-adjacent terms was passing.

  Three-layer fix:
    a) Raise _MIN_RELEVANCE from 0.15 → 0.22.
    b) Add a REQUIRED TECH HIT check: for tech queries, at least one of
       the HN boost words (all AI/tech terms) must appear in the title.
       "The Challenges of Implementing C++ Header Units" has zero boost
       words from the AI list → dropped before relevance scoring.
    c) Fix the fallback query: instead of stripping down to generic verbs
       ("challenges implementing"), extract the TOPIC NOUN from the original
       question and append it. "challenges implementing RAG" is fine;
       "challenges implementing" is not.

FIX 2 — Fallback query is now topic-noun-anchored.
  _extract_fallback_query() takes the first TECH-DOMAIN noun from the
  question (words in _TECH_QUERY_NOUNS set) and appends it to the fallback.
  If no tech noun found, skip the fallback entirely rather than sending a
  too-generic 2-word query.

FIX 3 — _score_hit() now returns 0.0 (not the raw relevance score) if no
  boost word hit AND the query is a tech query. This effectively requires
  at least one known AI/tech term in the HN title for tech questions.
  Non-tech queries (rare on HN) still use the raw relevance score.

All other logic unchanged from v3.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Set

import httpx

from core.logger import setup_logger
from app.retrieval.query_preprocessor import relevance_score, _STOP_WORDS

logger = setup_logger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

# FIX 1b: raised from 0.15 → 0.22
_MIN_RELEVANCE = 0.22
_MIN_POINTS    = 3
_MIN_COMMENTS  = 1
_MIN_TITLE_LEN = 10

# Tech term patterns (same as v3) — extract first match as primary HN query
_TECH_TERM_PATTERNS: List[tuple] = [
    ("diffusion model",         "diffusion models"),
    ("generative adversarial",  "GAN generative"),
    (" gan",                    "GAN generative"),
    ("gans",                    "GAN generative"),
    ("large language model",    "LLM large language model"),
    (" llm",                    "LLM large language"),
    ("retrieval augmented",     "RAG retrieval augmented"),
    (" rag ",                   "RAG retrieval"),
    ("fine-tun",                "fine-tuning LLM"),
    ("fine tuning",             "fine-tuning LLM"),
    ("transformer",             "transformer neural network"),
    ("stable diffusion",        "stable diffusion"),
    ("reinforcement learning",  "reinforcement learning RL"),
    ("computer vision",         "computer vision deep learning"),
    ("natural language",        "NLP natural language"),
    ("knowledge graph",         "knowledge graph"),
    ("vector database",         "vector database embeddings"),
    ("graph database",          "graph database"),
    ("embedding",               "embeddings vector"),
    ("neural network",          "neural network deep learning"),
    ("deep learning",           "deep learning"),
    ("machine learning",        "machine learning"),
    ("pytorch",                 "PyTorch"),
    ("tensorflow",              "TensorFlow"),
    ("openai",                  "OpenAI"),
    ("anthropic",               "Anthropic Claude"),
    ("huggingface",             "HuggingFace"),
    ("inference",               "ML inference"),
    ("benchmark",               "ML benchmark"),
    ("generative",              "generative AI"),
]

# Boost words — AI/ML/tech terms that must appear in title for tech queries
# FIX 1b: if NONE of these appear in the title AND query is tech → score=0.0
_HN_BOOST_WORDS: List[str] = [
    "diffusion", "gan", "gans", "generative", "llm", "gpt", "bert",
    "transformer", "embedding", "neural", "inference", "benchmark",
    "langchain", "llamaindex", "openai", "anthropic", "huggingface",
    "stable", "prompt", "agent", "chatgpt", "claude", "gemini",
    "training", "dataset", "model", "pytorch", "tensorflow",
    "deep learning", "machine learning", "reinforcement", "fine-tun",
    "rag", "retrieval", "vector", "attention", "autoregressive",
    "language model", "ai ", "ml ", " ai", " ml", "artificial intelligence",
    "knowledge graph", "graph database", "vector database",
]

# FIX 2: tech nouns — used to anchor the fallback query so it's never just
# generic verbs like "challenges implementing"
_TECH_QUERY_NOUNS: Set[str] = {
    "rag", "retrieval", "fine-tuning", "finetuning", "llm", "gpt", "bert",
    "transformer", "diffusion", "gan", "embedding", "neural", "inference",
    "machine", "learning", "language", "model", "attention", "vector",
    "database", "graph", "knowledge", "generation", "training",
}

# Signals that this is a tech/AI query — if any present, enforce boost-word gate
_TECH_QUERY_SIGNALS: Set[str] = {
    "rag", "retrieval", "fine-tuning", "finetuning", "llm", "gpt", "bert",
    "transformer", "embedding", "neural", "deep learning", "machine learning",
    "diffusion", "generative", "language model", "vector", "knowledge graph",
    "attention", "inference", "training", "dataset", "benchmark", "fine tuning",
}


def _is_tech_query(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _TECH_QUERY_SIGNALS)


def _title_has_boost_word(title: str) -> bool:
    """Return True if the title contains at least one AI/tech boost word."""
    t = title.lower()
    return any(kw in t for kw in _HN_BOOST_WORDS)


def _extract_hn_query(question: str) -> str:
    """
    Extract the best 1-3 word HN search query from a question.
    Uses known tech term patterns first, then fallback.
    """
    q = question.lower()
    for pattern, hn_query in _TECH_TERM_PATTERNS:
        if pattern in q:
            return hn_query
    # Generic fallback: first 2-3 content words
    words = re.findall(r"\b[\w-]+\b", q)
    content = [
        w for w in words
        if len(w) > 3
        and w not in _STOP_WORDS
        and w not in {"what", "how", "does", "between", "related",
                      "similarities", "differences", "limitations",
                      "compared", "tradeoffs", "advances", "directions",
                      "future", "primary", "recent", "industry", "terms",
                      "challenges", "implementing", "performance", "using"}
    ]
    return " ".join(content[:3]) if content else question[:40]


def _extract_fallback_query(question: str, primary_query: str) -> Optional[str]:
    """
    FIX 2: Build a fallback query that is anchored to a tech noun.

    Instead of taking the first 2 content words (which may be generic verbs
    like "challenges implementing"), find the first TECH noun in the question
    and use that as the anchor. If no tech noun found, return None so the
    fallback is skipped entirely.

    Examples:
      "What are the challenges in implementing RAG in real-world apps?"
      → fallback = "RAG" (first tech noun)
      → query = "RAG retrieval"

      "What are the performance differences between RAG and fine-tuning?"
      → fallback = "RAG fine-tuning"

    This prevents "challenges implementing" from going out naked.
    """
    q     = question.lower()
    words = re.findall(r"\b[\w-]+\b", q)

    tech_nouns = [w for w in words if w in _TECH_QUERY_NOUNS]
    if not tech_nouns:
        return None  # skip fallback entirely

    # Use up to 2 tech nouns as fallback
    fallback = " ".join(dict.fromkeys(tech_nouns[:2]))  # deduplicated, ordered

    if fallback == primary_query.lower():
        return None  # same as primary — nothing new

    return fallback


def _score_hit(question: str, title: str, content: str, is_tech: bool) -> float:
    """
    Score a HN hit.

    FIX 3: For tech queries, return 0.0 if no boost word appears in title.
    This prevents generic articles ("Challenges of Implementing X") from
    passing when X has nothing to do with AI/ML.
    """
    # FIX 3: tech query + no AI term in title → hard zero
    if is_tech and not _title_has_boost_word(title):
        return 0.0

    score = relevance_score(question, title, content)

    # Boost if a known tech term appears in title (single boost only)
    title_lower = title.lower()
    for kw in _HN_BOOST_WORDS:
        if kw in title_lower:
            score = min(score + 0.10, 1.0)
            break

    return score


async def _search_hn(
    client      : httpx.AsyncClient,
    query       : str,
    question    : str,
    is_tech     : bool,
    max_results : int,
    seen_urls   : Set[str],
) -> List[Dict]:
    """Run one HN Algolia search and return filtered results."""
    try:
        resp = await client.get(
            HN_SEARCH_URL,
            params={
                "query":       query,
                "tags":        "story",
                "hitsPerPage": max_results * 5,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"HackerNews search failed for '{query}': {e}")
        return []

    results: List[Dict] = []

    for hit in data.get("hits", []):
        if len(results) >= max_results:
            break

        title      = (hit.get("title") or "").strip()
        story_text = (hit.get("story_text") or "").strip()
        points     = int(hit.get("points") or 0)
        comments   = int(hit.get("num_comments") or 0)
        url        = (
            hit.get("url")
            or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        )

        if url in seen_urls:
            continue
        if len(title) < _MIN_TITLE_LEN:
            continue

        has_engagement = points >= _MIN_POINTS or comments >= _MIN_COMMENTS
        if not story_text and not has_engagement:
            continue

        content_parts = [title, f"Points: {points} | Comments: {comments}"]
        if story_text:
            content_parts.append(story_text[:1000])
        content = "\n".join(content_parts)

        # FIX 3: score_hit now enforces boost-word gate for tech queries
        score = _score_hit(question, title, content, is_tech)

        if score < _MIN_RELEVANCE:
            logger.debug(f"HN dropped (score={score:.2f}): '{title[:60]}'")
            continue

        seen_urls.add(url)
        results.append({
            "source":  "hackernews",
            "title":   title,
            "url":     url,
            "content": content,
            "score":   score,
        })

    return results


class HackerNewsRetriever:

    def __init__(self):
        logger.info("HackerNewsRetriever initialized")

    async def retrieve(self, question: str, max_results: int = 5) -> List[Dict]:
        primary_query = _extract_hn_query(question)
        is_tech       = _is_tech_query(question)
        logger.info(
            f"HackerNews | phrase='{primary_query}' "
            f"(from: '{question[:60]}') | is_tech={is_tech}"
        )

        seen_urls: Set[str] = set()
        results: List[Dict] = []

        async with httpx.AsyncClient(timeout=15) as client:
            # Pass 1: primary tech-term query
            results = await _search_hn(
                client, primary_query, question, is_tech, max_results, seen_urls
            )

            # Pass 2: topic-noun-anchored fallback if < 2 results
            if len(results) < 2:
                # FIX 2: fallback query is anchored to a tech noun, never bare verbs
                fallback_query = _extract_fallback_query(question, primary_query)

                if fallback_query:
                    logger.info(f"HackerNews | fallback phrase='{fallback_query}'")
                    extra = await _search_hn(
                        client, fallback_query, question, is_tech,
                        max_results - len(results), seen_urls,
                    )
                    results.extend(extra)
                else:
                    logger.debug("HackerNews | no valid fallback — skipping second pass")

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:max_results]

        logger.info(f"HackerNews returned {len(results)} results")
        return results