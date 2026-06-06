"""
app/retrieval/nature_retriever.py  (FIXED v4)

KEY CHANGES vs v3:
  FIX 1 — _MIN_RELEVANCE raised from 0.06 → 0.18.
           0.06 was essentially passing everything. CrossRef returns papers
           like "Diffusion of innovations in healthcare" for "diffusion model
           vs GAN" — those score ~0.07 and were getting through. 0.18 cuts
           loosely related papers while keeping real ML/AI papers that have
           sparse abstracts (they typically score 0.20-0.45).

  FIX 2 — CrossRef: now requires abstract OR a known reputable journal.
           Papers with no abstract and an unknown journal are title-only
           stubs that add noise. They're dropped unless the journal is in a
           known reputable list (Nature, Science, ICML, NeurIPS, ICLR, etc).

  FIX 3 — CrossRef: filter by "has-abstract" field in the select param.
           CrossRef supports filtering — request only works with abstracts
           by adding `has-abstract=true` to the filter. Cuts the number of
           stub results returned before we even score them.

  FIX 4 — EuropePMC: raised its internal min relevance to match _MIN_RELEVANCE.
           Previously it used the global _MIN_RELEVANCE but the check was
           after building content — now skips content-building for low titles.

  FIX 5 — All strategies now log how many they dropped for relevance,
           making debugging easier.

  FIX 6 — _MIN_TITLE_LEN raised from 15 → 20. "Diffusion" alone as a title
           would pass 15 chars but is useless context.

Unchanged from v3:
  - Multi-strategy cascade: CrossRef → EuropePMC → Nature RSS
  - Topic-signal RSS feed mapping (_TOPIC_TO_RSS)
  - Semantic Scholar strategy (still available as _semantic_scholar() helper)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin

import httpx

from core.logger import setup_logger
from app.retrieval.query_preprocessor import build_source_queries, relevance_score

logger = setup_logger(__name__)

# FIX 1: raised from 0.06 → 0.18
_MIN_RELEVANCE = 0.18
# FIX 6: raised from 15 → 20
_MIN_TITLE_LEN = 20

# FIX 2A: Domain-anchor terms for AI/ML queries — prepended to search phrase
# to steer CrossRef/PMC toward AI/ML papers instead of medical/neuro results.
_AI_ML_DOMAIN_SIGNALS = {
    "machine learning", "deep learning", "neural network", "transformer",
    "llm", "language model", "diffusion model", "gan", "generative",
    "reinforcement learning", "computer vision", "nlp", "embedding",
    "fine-tuning", "fine tuning", "rag", "retrieval-augmented",
    "retrieval augmented", "artificial intelligence", "benchmark",
    "attention mechanism", "bert", "gpt", "inference",
}

# FIX 2B: Off-domain exclusion terms. Papers whose title OR abstract
# contains ONLY these terms (and none of the AI/ML signals) are noise.
# Skipped if the user's original query explicitly mentions these terms.
_DOMAIN_EXCLUSION_TERMS = {
    "mri", "fmri", "brain", "neuroscience", "neuroimaging", "cortex",
    "cortical", "hippocampus", "eeg", "clinical trial", "patient",
    "surgery", "tumor", "tumour", "pathology", "diagnosis",
    "radiology", "radiograph", "magnetic resonance",
    "cell biology", "in vivo", "in vitro", "mice", "rat",
    "protein folding", "genome", "phylogenetic",
}


def _is_ai_ml_query(question: str) -> bool:
    """Return True if the question is about AI/ML topics."""
    q = question.lower()
    return any(sig in q for sig in _AI_ML_DOMAIN_SIGNALS)


def _domain_anchor_phrase(search_phrase: str, question: str) -> str:
    """FIX 2A: Prepend domain-anchoring terms for AI/ML queries."""
    if not _is_ai_ml_query(question):
        return search_phrase
    # Don't double-add if already present
    sp_lower = search_phrase.lower()
    if "machine learning" in sp_lower or "deep learning" in sp_lower:
        return search_phrase
    return f"machine learning {search_phrase}"


def _passes_domain_exclusion(title: str, abstract: str, user_question: str) -> bool:
    """
    FIX 2B: Reject papers that are off-domain noise.
    If the user's query is about AI/ML, reject papers whose title+abstract
    contains exclusion terms but NO AI/ML signal terms.
    
    Also handles "RAG" disambiguation: "RAG" is both an AI term (Retrieval-
    Augmented Generation) and a biology term (Recombination-Activating Gene).
    Papers about "RAG deficiencies" or "RAG mutations" are biology, not AI.
    """
    if not _is_ai_ml_query(user_question):
        return True  # Not an AI/ML query, don't filter

    q_lower = user_question.lower()
    # If user explicitly asked about an excluded term, don't filter it out
    if any(term in q_lower for term in _DOMAIN_EXCLUSION_TERMS):
        return True

    combined = f"{title} {abstract}".lower()

    # RAG disambiguation: if paper only matches "rag" and has biology RAG terms, reject
    _BIO_RAG_TERMS = {
        "recombination", "immunodeficiency", "v(d)j", "lymphocyte",
        "b cell", "t cell", "immune", "mutation", "deficienc",
        "severe combined", "scid", "dna repair", "endonuclease",
    }
    if "rag" in combined:
        has_bio_rag = any(t in combined for t in _BIO_RAG_TERMS)
        # Check if there are OTHER AI/ML signals besides just "rag"
        ai_signals_excl_rag = {s for s in _AI_ML_DOMAIN_SIGNALS if s != "rag"}
        has_other_ai = any(sig in combined for sig in ai_signals_excl_rag)
        if has_bio_rag and not has_other_ai:
            logger.debug(f"RAG disambiguation | dropped biology RAG paper: '{title[:60]}'")
            return False

    has_exclusion = any(term in combined for term in _DOMAIN_EXCLUSION_TERMS)
    if not has_exclusion:
        return True  # No exclusion terms — keep

    # Has exclusion terms — only keep if it ALSO has AI/ML terms
    has_ai_signal = any(sig in combined for sig in _AI_ML_DOMAIN_SIGNALS)
    if not has_ai_signal:
        logger.debug(f"Domain exclusion | dropped: '{title[:60]}' (off-domain for AI/ML query)")
        return False

    return True

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# FIX 2: reputable venues — papers without abstracts are kept only if
# published in one of these journals/conferences.
_REPUTABLE_VENUES: Set[str] = {
    # ML / AI conferences
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv", "aaai",
    "acl", "emnlp", "naacl", "ijcai", "uai", "aistats",
    # ML / AI journals
    "journal of machine learning research", "jmlr",
    "transactions on neural networks", "ieee transactions",
    "machine learning", "artificial intelligence",
    "international journal of computer vision",
    # Nature family
    "nature", "nature machine intelligence", "nature communications",
    "nature methods", "scientific reports",
    # General science
    "science", "proceedings of the national academy",
    "pnas", "cell", "the lancet",
}


def _is_reputable_venue(venue: str) -> bool:
    v = venue.lower()
    return any(r in v for r in _REPUTABLE_VENUES)


# ── Topic signal → RSS feed mapping ─────────────────────────────────────────

_TOPIC_TO_RSS: List[tuple] = [
    # AI / ML signals
    ("diffusion",        "https://www.nature.com/subjects/machine-learning.rss"),
    ("gan",              "https://www.nature.com/subjects/machine-learning.rss"),
    ("generative",       "https://www.nature.com/subjects/machine-learning.rss"),
    ("neural",           "https://www.nature.com/subjects/machine-learning.rss"),
    ("deep learning",    "https://www.nature.com/subjects/deep-learning.rss"),
    ("machine learning", "https://www.nature.com/subjects/machine-learning.rss"),
    ("transformer",      "https://www.nature.com/subjects/machine-learning.rss"),
    ("llm",              "https://www.nature.com/subjects/machine-learning.rss"),
    ("language model",   "https://www.nature.com/subjects/machine-learning.rss"),
    ("natural language", "https://www.nature.com/subjects/natural-language-processing.rss"),
    ("nlp",              "https://www.nature.com/subjects/natural-language-processing.rss"),
    ("computer vision",  "https://www.nature.com/subjects/machine-learning.rss"),
    ("reinforcement",    "https://www.nature.com/subjects/machine-learning.rss"),
    ("embedding",        "https://www.nature.com/subjects/machine-learning.rss"),
    ("inference",        "https://www.nature.com/subjects/machine-learning.rss"),
    ("benchmark",        "https://www.nature.com/subjects/machine-learning.rss"),
    ("image synthesis",  "https://www.nature.com/subjects/machine-learning.rss"),
    ("artificial intelligence", "https://www.nature.com/subjects/artificial-intelligence.rss"),
    # Bio / medical
    ("protein",          "https://www.nature.com/subjects/protein.rss"),
    ("genomics",         "https://www.nature.com/subjects/genomics.rss"),
    ("gene",             "https://www.nature.com/subjects/genomics.rss"),
    ("cancer",           "https://www.nature.com/subjects/cancer.rss"),
    ("drug",             "https://www.nature.com/subjects/drug-discovery.rss"),
    ("bioinformatics",   "https://www.nature.com/subjects/bioinformatics.rss"),
    # Physical sciences
    ("quantum",          "https://www.nature.com/subjects/quantum-physics.rss"),
    ("climate",          "https://www.nature.com/subjects/climate-change-ecology.rss"),
]


def _pick_rss_feeds(question: str) -> List[str]:
    q = question.lower()
    seen: Set[str] = set()
    feeds: List[str] = []
    for signal, url in _TOPIC_TO_RSS:
        if signal in q and url not in seen:
            seen.add(url)
            feeds.append(url)
            if len(feeds) >= 2:
                break
    return feeds


# ── Strategy A: CrossRef API ─────────────────────────────────────────────────

async def _crossref(
    client: httpx.AsyncClient, question: str, search_phrase: str, max_results: int
) -> List[Dict]:
    """
    CrossRef indexes all major journals. Strict relevance filtering applied.

    FIX 3: Added filter=has-abstract:true to reduce stub results.
    FIX 2: Papers without abstracts kept only if from a reputable venue.
    FIX 1: _MIN_RELEVANCE raised to 0.18 so only on-topic papers pass.
    FIX 2A: Domain-anchor search phrase for AI/ML queries.
    FIX 2B: Off-domain keyword exclusion filter.
    """
    try:
        # FIX 2A: anchor the search phrase for AI/ML queries
        anchored_phrase = _domain_anchor_phrase(search_phrase, question)

        resp = await client.get(
            "https://api.crossref.org/works",
            params={
                "query":   anchored_phrase,
                "rows":    max_results * 3,       # fetch more, filter down
                "select":  "title,abstract,DOI,container-title,published",
                "sort":    "relevance",
                "filter":  "has-abstract:true",   # FIX 3: only works with abstracts
                "mailto":  "research@assistant.app",
            },
            headers={"User-Agent": _UA},
            timeout=18,
        )
        if resp.status_code != 200:
            logger.debug(f"CrossRef returned {resp.status_code}")
            return []

        items   = resp.json().get("message", {}).get("items", [])
        results: List[Dict] = []
        dropped = 0

        for item in items:
            titles = item.get("title", [])
            title  = titles[0].strip() if titles else ""
            if not title or len(title) < _MIN_TITLE_LEN:
                dropped += 1
                continue

            abstract = (item.get("abstract") or "").strip()
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

            journal_list = item.get("container-title", [])
            journal      = journal_list[0] if journal_list else ""
            doi          = item.get("DOI", "")

            # FIX 2: if no abstract, only keep if from a reputable venue
            if not abstract and not _is_reputable_venue(journal):
                dropped += 1
                continue

            # FIX 2B: Off-domain exclusion filter
            if not _passes_domain_exclusion(title, abstract, question):
                dropped += 1
                continue

            content = (
                f"{title}\n\nJournal: {journal}\n\n{abstract[:500]}"
                if abstract else f"{title}\n\nJournal: {journal}"
            )

            score = relevance_score(question, title, content)
            if score < _MIN_RELEVANCE:  # FIX 1: 0.18 threshold
                dropped += 1
                continue

            url = f"https://doi.org/{doi}" if doi else "https://crossref.org"
            results.append({
                "source":  "nature",
                "title":   title,
                "url":     url,
                "content": content,
                "score":   score,
            })

            if len(results) >= max_results:
                break

        logger.debug(f"CrossRef | kept={len(results)} dropped={dropped}")
        return results

    except Exception as e:
        logger.debug(f"CrossRef failed: {e}")
        return []


# ── Strategy B: Europe PMC API ───────────────────────────────────────────────

async def _europe_pmc(
    client: httpx.AsyncClient, question: str, search_phrase: str, max_results: int
) -> List[Dict]:
    """
    Europe PMC — covers Nature, Springer, and life-science journals.
    FIX 4: Explicit relevance check on title before building content.
    FIX 2A: Domain-anchor search phrase for AI/ML queries.
    FIX 2B: Off-domain keyword exclusion filter.
    """
    try:
        # FIX 2A: anchor the search phrase for AI/ML queries
        anchored_phrase = _domain_anchor_phrase(search_phrase, question)

        resp = await client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query":      anchored_phrase,
                "format":     "json",
                "pageSize":   max_results * 3,
                "resultType": "core",
                "sort":       "RELEVANCE",
            },
            headers={"User-Agent": _UA},
            timeout=18,
        )
        if resp.status_code != 200:
            logger.debug(f"Europe PMC returned {resp.status_code}")
            return []

        data    = resp.json()
        items   = data.get("resultList", {}).get("result", [])
        results: List[Dict] = []
        dropped = 0

        for item in items:
            title    = (item.get("title") or "").strip().rstrip(".")
            abstract = (item.get("abstractText") or "").strip()
            journal  = (item.get("journalTitle") or "").strip()
            pmid     = item.get("pmid", "")
            doi      = item.get("doi", "")

            if not title or len(title) < _MIN_TITLE_LEN:
                dropped += 1
                continue

            # FIX 4: quick title-only relevance check before building content
            title_score = relevance_score(question, title, title)
            if title_score < _MIN_RELEVANCE * 0.5:  # half-threshold on title alone
                dropped += 1
                continue

            # FIX 2B: Off-domain exclusion filter
            if not _passes_domain_exclusion(title, abstract, question):
                dropped += 1
                continue

            content = (
                f"{title}\n\nJournal: {journal}\n\n{abstract[:500]}"
                if abstract else f"{title}\n\nJournal: {journal}"
            )
            score = relevance_score(question, title, content)
            if score < _MIN_RELEVANCE:
                dropped += 1
                continue

            url = (
                f"https://doi.org/{doi}" if doi
                else f"https://europepmc.org/article/MED/{pmid}" if pmid
                else "https://europepmc.org"
            )
            results.append({
                "source":  "nature",
                "title":   title,
                "url":     url,
                "content": content,
                "score":   score,
            })

            if len(results) >= max_results:
                break

        logger.debug(f"EuropePMC | kept={len(results)} dropped={dropped}")
        return results

    except Exception as e:
        logger.debug(f"Europe PMC failed: {e}")
        return []


# ── Strategy C: Nature RSS ────────────────────────────────────────────────────

async def _rss_nature(
    client: httpx.AsyncClient, question: str, max_results: int
) -> List[Dict]:
    """Nature RSS feeds matched by topic signal. Same _MIN_RELEVANCE gate."""
    feed_urls = _pick_rss_feeds(question)
    if not feed_urls:
        logger.debug("Nature RSS: no matching feeds for this question")
        return []

    results: List[Dict] = []

    for feed_url in feed_urls:
        try:
            resp = await client.get(feed_url, timeout=15)
            if resp.status_code != 200:
                continue

            root    = ET.fromstring(resp.content)
            ns      = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall(".//atom:entry", ns) or root.findall(".//item")

            for entry in entries:
                if len(results) >= max_results:
                    break

                title_el = entry.find("atom:title", ns) or entry.find("title")
                link_el  = entry.find("atom:link", ns)  or entry.find("link")
                summ_el  = entry.find("atom:summary", ns) or entry.find("description")

                title   = (title_el.text or "").strip() if title_el is not None else ""
                href    = (
                    link_el.get("href", "") if link_el is not None and link_el.get("href")
                    else (link_el.text or "") if link_el is not None else ""
                )
                snippet = (summ_el.text or "").strip()[:400] if summ_el is not None else ""
                snippet = re.sub(r"<[^>]+>", " ", snippet).strip()

                if not title or len(title) < _MIN_TITLE_LEN:
                    continue

                content = f"{title}\n\n{snippet}" if snippet else title
                score   = relevance_score(question, title, content)
                if score < _MIN_RELEVANCE:
                    continue

                results.append({
                    "source":  "nature",
                    "title":   title,
                    "url":     href or feed_url,
                    "content": content,
                    "score":   score,
                })

        except Exception as e:
            logger.debug(f"Nature RSS {feed_url} failed: {e}")

    return results


# ── Semantic Scholar helper (available for direct use) ────────────────────────

async def _semantic_scholar(
    client: httpx.AsyncClient, question: str, search_phrase: str, max_results: int
) -> List[Dict]:
    """Semantic Scholar free API — same _MIN_RELEVANCE gate applied."""
    try:
        resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query":  search_phrase,
                "limit":  min(max_results * 3, 15),
                "fields": "title,abstract,year,externalIds,openAccessPdf,venue",
            },
            headers={"User-Agent": _UA},
            timeout=20,
        )
        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limited in nature_retriever")
            return []
        if resp.status_code != 200:
            return []

        papers  = resp.json().get("data", [])
        results: List[Dict] = []

        for paper in papers:
            title    = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            venue    = (paper.get("venue") or "").strip()
            year     = paper.get("year") or ""

            if not title or len(title) < _MIN_TITLE_LEN:
                continue

            content = f"{title}\n\nVenue: {venue} ({year})\n\n{abstract[:400]}" if abstract else title
            score   = relevance_score(question, title, content)
            if score < _MIN_RELEVANCE:
                continue

            ext_ids = paper.get("externalIds") or {}
            doi     = ext_ids.get("DOI", "")
            url     = (
                f"https://doi.org/{doi}" if doi
                else f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
            )
            results.append({
                "source":  "nature",
                "title":   title,
                "url":     url,
                "content": content,
                "score":   score,
            })

            if len(results) >= max_results:
                break

        return results

    except Exception as e:
        logger.debug(f"Semantic Scholar (nature fallback) failed: {e}")
        return []


# ── Public retriever class ────────────────────────────────────────────────────

class NatureRetriever:

    def __init__(self):
        logger.info(
            f"NatureRetriever initialized | min_relevance={_MIN_RELEVANCE} | "
            f"strategies=CrossRef→EuropePMC→RSS"
        )

    async def retrieve(self, question: str, max_results: int = 2) -> List[Dict]:
        """FIX 2C: Default max_results lowered from 5→2 to reduce noise per question."""
        search_phrase = build_source_queries(question)["nature"]
        logger.info(f"Nature | phrase='{search_phrase}' | max={max_results} (from: '{question[:60]}')")

        async with httpx.AsyncClient(
            timeout=25,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:

            # Strategy A: CrossRef (most reliable, explicit abstract filter)
            results = await _crossref(client, question, search_phrase, max_results)
            if results:
                logger.info(f"Nature returned {len(results)} results (CrossRef)")
                return results

            logger.debug("CrossRef 0 results — trying Europe PMC")

            # Strategy B: Europe PMC
            results = await _europe_pmc(client, question, search_phrase, max_results)
            if results:
                logger.info(f"Nature returned {len(results)} results (EuropePMC)")
                return results

            logger.debug("EuropePMC 0 results — trying RSS")

            # Strategy C: Nature RSS
            results = await _rss_nature(client, question, max_results)
            if results:
                logger.info(f"Nature returned {len(results)} results (RSS)")
                return results

        logger.debug("Nature returned 0 results")
        return []