from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = unicodedata.normalize("NFKC", text)
    return text


# ---------------------------------------------------------------------------
# Filler patterns
# ---------------------------------------------------------------------------
# Comprehensive front-filler removal.
# Strips question openers like "How does", "What are the key", etc.
# Uses a two-pass approach: first strip the opener phrase, then strip
# any dangling articles/adjectives left at the start.
_FRONT_FILLERS = re.compile(
    r"^(?:"
    # "How does/do/did/is/are/can X" patterns
    r"how\s+(?:does|do|did|is|are|can|were)\s+(?:the\s+|a\s+|an\s+)?|"
    # "What is/are/were/was the? (adj)?" patterns
    r"what\s+(?:is|are|were|was|have|has)\s+(?:the\s+)?(?:core\s+|primary\s+|fundamental\s+|key\s+|main\s+|common\s+|some\s+|recent\s+)?|"
    # "What are the? (adj)?" without second verb
    r"what\s+are\s+(?:the\s+)?(?:primary\s+|key\s+|main\s+|common\s+|some\s+|typical\s+|recent\s+)?|"
    # "What recent advances have been made in the development of"
    r"what\s+recent\s+advances?\s+have\s+been\s+made\s+in\s+(?:the\s+development\s+of\s+)?|"
    # "In what ways/scenarios/cases"
    r"in\s+what\s+(?:ways?|scenarios?|cases?)\s+|"
    # "Why is/are/do/does"
    r"why\s+(?:is|are|do|does)\s+|"
    # "When is/are/do/does/did"
    r"when\s+(?:is|are|do|does|did)\s+|"
    # "Which is/are"
    r"which\s+(?:is|are)\s+|"
    # "Describe/Explain/Compare the/how/why"
    r"describe\s+(?:the\s+|how\s+|why\s+)?|"
    r"explain\s+(?:the\s+|how\s+|why\s+)?|"
    r"compare\s+(?:the\s+)?"
    r")",
    re.IGNORECASE,
)

# Second pass: strip dangling leading articles/adjectives after filler removal
_LEADING_JUNK = re.compile(
    r"^(?:the|a|an|its|their|these|those|such|es|s)\s+",
    re.IGNORECASE,
)

_TAIL_FILLERS = re.compile(
    r"\s*(in\s+terms\s+of\s+[\w\s]+|"
    r"for\s+(?:a\s+)?(?:developer|researcher|user|team)|"
    r"compared\s+to\s+(?:those\s+of\s+)?|"
    r"between\s+(?:the\s+two|them)|"
    r"\?)$",
    re.IGNORECASE,
)

_POSSESSIVE = re.compile(r"'s\b", re.IGNORECASE)

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "vs", "versus",
    "for", "to", "is", "are", "was", "were", "how", "what", "why",
    "do", "does", "between", "on", "at", "with", "from", "that",
    "this", "these", "those", "have", "has", "been", "being",
    "would", "could", "should", "will", "can", "may", "might",
    "over", "under", "about", "around", "into", "through",
    "some", "any", "each", "every", "both", "all", "more", "most",
    "its", "their", "our", "your", "my", "his", "her",
}

_CONNECTIVES = {"and", "or", "vs", "versus", "with", "without"}

_DOMAIN_HINTS: Dict[str, str] = {
    "vector database":      "vector database",
    "graph database":       "graph database",
    "relational database":  "relational database",
    "llm":                  "large language model",
    "large language":       "large language model",
    "neural network":       "neural network",
    "deep learning":        "deep learning",
    "machine learning":     "machine learning",
    "transformer":          "transformer model",
    "embedding":            "vector embedding",
    "rag":                  "retrieval augmented generation",
    "reinforcement":        "reinforcement learning",
    "computer vision":      "computer vision",
    "diffusion":            "diffusion model",
    "natural language":     "natural language processing",
    "knowledge graph":      "knowledge graph",
}


def _strip_fillers(text: str) -> str:
    text = text.strip().rstrip("?").strip()
    text = _POSSESSIVE.sub(" ", text).strip()
    text = _FRONT_FILLERS.sub("", text).strip()
    # Second pass: remove any dangling article/fragment left by filler removal
    # e.g. "es the inference speed" → "inference speed"
    text = _LEADING_JUNK.sub("", text).strip()
    text = _TAIL_FILLERS.sub("", text).strip()
    return text


def _extract_keywords(text: str, max_words: int = 6) -> str:
    words = text.split()
    kept: List[str] = []

    i = 0
    while i < len(words) and len(kept) < max_words:
        w = words[i].lower().strip(",'\"")

        if w not in _STOP_WORDS or w in _CONNECTIVES:
            if i + 1 < len(words):
                bigram = f"{w} {words[i+1].lower().strip(',')}"
                if any(bigram in hint for hint in _DOMAIN_HINTS):
                    kept.append(words[i])
                    kept.append(words[i + 1])
                    i += 2
                    continue
            kept.append(words[i])
        i += 1

    return " ".join(kept)


def _add_domain_hint(phrase: str, original_question: str) -> str:
    """
    Append domain hint ONLY if the phrase is short (≤3 words).
    Long phrases already have enough context; appending hints bloats them
    and breaks Wikipedia title matching.
    """
    if len(phrase.split()) > 3:
        return phrase

    q_lower      = original_question.lower()
    phrase_lower = phrase.lower()

    for signal, hint in _DOMAIN_HINTS.items():
        if signal in q_lower and hint not in phrase_lower:
            if not any(part in phrase_lower for part in hint.split()):
                return f"{phrase} {hint}".strip()
            return phrase

    return phrase


def _clean(phrase: str) -> str:
    words = phrase.split()
    deduped: List[str] = []
    prev = None
    for w in words:
        if w.lower() != (prev or "").lower():
            deduped.append(w)
        prev = w
    return " ".join(deduped).strip()


def _wiki_phrase(question: str) -> str:
    """
    Build a very short (2-3 word) phrase for Wikipedia title search.
    Wikipedia opensearch is title-prefix based — shorter is better.

    Strategy: extract ONLY non-stop, non-connective content words from the
    stripped question, then take the first 3.  This gives clean noun phrases
    like "Diffusion models" or "inference speed diffusion" that actually
    match Wikipedia article titles.
    """
    import re as _re
    stripped = _strip_fillers(question)

    _ALL_JUNK = _STOP_WORDS | {"and", "or", "vs", "versus", "with", "without",
                                "related", "using", "used", "between", "differ",
                                "compare", "compared", "typical", "primary",
                                "key", "core", "main", "common", "recent",
                                "similarities", "differences", "limitations",
                                "strengths", "advances", "benchmarks"}

    raw_words = _re.findall(r"\b[\w-]+\b", stripped)
    content_words = [w for w in raw_words if len(w) > 2 and w.lower() not in _ALL_JUNK]

    if not content_words:
        # Fallback: just use the stripped question, capped at 3 tokens
        return _clean(" ".join(stripped.split()[:3]))

    return _clean(" ".join(content_words[:3]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_search_phrase(question: str, max_words: int = 6) -> str:
    """
    Convert a planner question to a short search phrase suitable for most APIs.
    """
    question   = _normalize_text(question)
    stripped   = _strip_fillers(question)
    keywords   = _extract_keywords(stripped, max_words=max_words)
    with_hint  = _add_domain_hint(keywords, question)
    return _clean(with_hint)


def build_source_queries(question: str) -> Dict[str, str]:
    """
    Build a per-source optimised query from a planner question.

    Source-specific decisions
    -------------------------
    tavily           — full question (semantic search engine, handles it best)
    wikipedia        — 2-3 word noun phrase, NO domain hints (title-prefix match)
    arxiv            — 4-5 keyword academic phrase, WITH hint if short
    semantic_scholar — same as arxiv
    pubmed           — 5-6 word medical phrase
    hackernews       — 3-4 word tech phrase, NO hint suffix
    nature           — 4-5 word science phrase, NO hint suffix
    mcp              — 5-6 word base phrase
    """
    question = _normalize_text(question)

    stripped  = _strip_fillers(question)
    base      = _clean(_extract_keywords(stripped, max_words=6))
    academic  = _clean(_extract_keywords(stripped, max_words=5))
    pubmed    = _clean(_extract_keywords(stripped, max_words=5))
    hn_phrase = _clean(_extract_keywords(stripped, max_words=4))  # no hint
    nat_phrase= _clean(_extract_keywords(stripped, max_words=5))  # no hint

    # Add domain hint only to academic sources (and only when phrase is short)
    academic_with_hint  = _add_domain_hint(academic,  question)
    base_with_hint      = _add_domain_hint(base,      question)
    pubmed_with_hint    = _add_domain_hint(pubmed,    question)

    return {
        "tavily":           question,                    # full question
        "wikipedia":        _wiki_phrase(question),      # 2-3 words, no hints
        "arxiv":            academic_with_hint,          # 4-5 words + hint
        "semantic_scholar": academic_with_hint,          # 4-5 words + hint
        "pubmed":           pubmed_with_hint,           # 5-6 words + hint
        "hackernews":       hn_phrase,                   # 3-4 words, no hint
        "nature":           nat_phrase,                  # 3-4 words, no hint
        "mcp":              base_with_hint,              # 5-6 words + hint
    }


# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------

def relevance_score(question: str, title: str, content: str) -> float:
    """
    Compute a 0-1 relevance score between a planner question and a document.

    The score combines keyword overlap with a small phrase-level bonus so that
    title-prefix matches and near-exact topic matches are not undervalued.
    """
    question = _normalize_text(question)
    stripped = _strip_fillers(question)

    q_terms   = set(_extract_keywords(stripped, max_words=10).lower().split())
    raw_terms = set(
        w.lower()
        for w in re.findall(r"\b[\w-]+\b", question)
        if len(w) > 3 and w.lower() not in _STOP_WORDS
    )
    terms = q_terms | raw_terms

    if not terms:
        return 0.5

    title_lower   = title.lower()
    content_lower = content[:800].lower()

    title_hits   = sum(1 for t in terms if t in title_lower)
    content_hits = sum(1 for t in terms if t in content_lower)

    score = (title_hits * 3 + content_hits) / (len(terms) * 4)

    # Phrase-level bonus: helps short topic phrases such as "diffusion models"
    # or "retrieval augmented generation" rank highly when titles match closely.
    phrase = " ".join(stripped.split()[:5]).lower().strip()
    if phrase:
        if phrase in title_lower:
            score += 0.20
        elif phrase in content_lower:
            score += 0.10

    # Bonus for exact title-topic overlap when the question already looks compact.
    compact_terms = [w.lower() for w in stripped.split() if w.lower() not in _STOP_WORDS]
    compact_phrase = " ".join(compact_terms[:4]).strip()
    if compact_phrase and compact_phrase in title_lower:
        score += 0.10

    return round(min(score, 1.0), 3)


def is_relevant(question: str, title: str, content: str, threshold: float = 0.15) -> bool:
    return relevance_score(question, title, content) >= threshold


# ---------------------------------------------------------------------------
# Query-type classifiers — used by HybridRetriever._plan_sources()
# ---------------------------------------------------------------------------

_TECH_SIGNALS = {
    # AI / ML
    "rag", "retrieval", "fine-tuning", "finetuning", "fine tuning",
    "llm", "gpt", "bert", "transformer", "embedding", "vector",
    "neural", "deep learning", "machine learning", "diffusion",
    "reinforcement", "computer vision", "natural language", "nlp",
    "langchain", "llamaindex", "hugging face", "huggingface",
    "openai", "anthropic", "claude", "chatgpt", "gemini",
    "inference", "training", "dataset", "benchmark", "attention",
    "quantization", "distillation", "knowledge graph", "graph database",
    "vector database",
    # Software / infra
    "api", "rest", "graphql", "microservice", "kubernetes", "docker",
    "serverless", "cloud", "database", "sql", "nosql", "redis",
    "kafka", "streaming", "pipeline", "ci/cd", "devops", "mlops",
    "python", "javascript", "typescript", "rust", "golang",
    "react", "fastapi", "pytorch", "tensorflow", "jax",
    # General tech
    "algorithm", "architecture", "scalability", "performance",
    "latency", "throughput", "open source", "github",
}

_SCIENCE_SIGNALS = {
    # Life sciences
    "biology", "gene", "protein", "cell", "tissue", "organ",
    "mutation", "dna", "rna", "crispr", "biomarker", "epidemiology",
    "genomics", "proteomics", "evolution", "species", "ecology",
    "neuroscience", "cognitive", "psychology", "behavior",
    # Physical / earth sciences
    "physics", "chemistry", "quantum", "particle", "atom", "molecule",
    "thermodynamics", "photon", "laser", "semiconductor", "superconductor",
    "climate", "fossil", "geology", "astronomy", "cosmology",
    "galaxy", "planet",
    # Research signals
    "experiment", "laboratory", "hypothesis", "peer-reviewed",
    "journal", "findings", "evidence", "study", "trial",
}

_MEDICAL_SIGNALS = {
    "disease", "drug", "medicine", "health", "cancer", "virus",
    "bacteria", "treatment", "clinical", "patient", "medical",
    "vaccine", "therapy", "diagnosis", "symptom", "hospital",
    "covid", "infection", "disorder", "syndrome", "pharmaceutical",
    "neurological", "cardiac", "pubmed", "ncbi",
}

_ACADEMIC_SIGNALS = (
    _TECH_SIGNALS
    | _SCIENCE_SIGNALS
    | _MEDICAL_SIGNALS
    | {
        "paper", "arxiv", "survey", "review", "model", "method",
        "approach", "framework", "evaluation", "accuracy", "precision",
        "recall", "f1", "metric", "ablation", "sota",
    }
)


def is_tech_query(question: str) -> bool:
    """True if the question is about technology / AI / software."""
    q = question.lower()
    return any(signal in q for signal in _TECH_SIGNALS)


def is_science_query(question: str) -> bool:
    """True if the question is about natural / life / physical science."""
    q = question.lower()
    return any(signal in q for signal in _SCIENCE_SIGNALS)


def is_medical_query(question: str) -> bool:
    """True if the question is about medicine / health / clinical topics."""
    q = question.lower()
    return any(signal in q for signal in _MEDICAL_SIGNALS)


def is_academic_query(question: str) -> bool:
    """True if the question is likely to have academic paper coverage."""
    q = question.lower()
    return any(signal in q for signal in _ACADEMIC_SIGNALS)