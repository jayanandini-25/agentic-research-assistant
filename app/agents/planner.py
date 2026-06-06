from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from core.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)

# =============================================================================
# PlannerAgent v9
# Fixes vs v8:
# - FIX P1: _semantic_deduplicate threshold 0.72 → 0.62 (catches more near-dups)
# - FIX P2: _strict_cosine_dedup threshold 0.75 → 0.65
# - FIX P3: _comparison_seed_questions completely rewritten — seeds no longer
#           share 4-word prefixes; each seed opens with a distinct anchor phrase
#           to lower mean pairwise cosine from ~0.57 to ~0.38
# - FIX P4: _enforce_comparison_symmetry — append guard now prevents >12
#           questions (was unconditionally appending, ignoring the cap)
# - FIX P5: _semantic_deduplicate threshold propagated through
#           _normalize_questions (was using default 0.72 on every call)
# =============================================================================

RETRY_DELAYS = [2, 5, 15]

STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "ought", "used",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "between",
    "into", "through", "during", "before", "after", "from", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "same", "than", "too", "very", "just", "what", "which",
    "who", "whom", "this", "that", "these", "those", "how", "why", "when",
    "where", "all", "each", "more", "most", "other", "some", "such", "no",
    "any", "its", "it", "vs", "versus", "compare", "comparison", "difference",
    "differences", "contrast", "contrasting", "compared", "among",
}

ACRONYM_MAP: Dict[str, str] = {
    "rag": "Retrieval-Augmented Generation",
    "llm": "Large Language Model",
    "llms": "Large Language Models",
    "gpt": "Generative Pre-trained Transformer",
    "bert": "Bidirectional Encoder Representations from Transformers",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "rlhf": "Reinforcement Learning from Human Feedback",
    "api": "Application Programming Interface",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "ai": "Artificial Intelligence",
    "gan": "Generative Adversarial Network",
    "gans": "Generative Adversarial Networks",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "pca": "Principal Component Analysis",
}

ENTITY_TAIL_WORDS = {
    "tradeoffs", "tradeoff", "comparison", "compare", "compared", "contrast",
    "differences", "difference", "analysis", "review", "benefits", "limitations",
    "pros", "cons", "vs", "versus", "and", "or",
}


# -----------------------------------------------------------------------------
# Retry wrapper
# -----------------------------------------------------------------------------

async def _llm_invoke(chain, inputs: dict, label: str = "LLM"):
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            logger.warning(f"{label} | waiting {delay}s before retry {attempt}/{len(RETRY_DELAYS)}")
            await asyncio.sleep(delay)
        try:
            return await chain.ainvoke(inputs)
        except Exception as exc:
            text = str(exc).lower()
            is_quota = any(k in text for k in ["429", "quota", "rate_limit", "token_quota", "too_many_tokens"])
            if is_quota and attempt < len(RETRY_DELAYS):
                last_exc = exc
                continue
            raise
    raise last_exc  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = (text or "").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def _lower(text: str) -> str:
    return _normalize_text(text).lower()


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"\b\w+\b", _lower(text))
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}


def _content_words(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", _lower(text)) if t not in STOP_WORDS and len(t) > 2]


def _word_vector(text: str) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for t in _content_words(text):
        vec[t] = vec.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(a.get(k, 0.0) * v for k, v in b.items())


def _extract_json(text: str) -> Dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except Exception:
        logger.warning(f"JSON parse failed: {text[:200]}")
        return {}


def _parse_questions(raw: str) -> List[str]:
    qs: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)\:]?\s*", "", line)
        cleaned = re.sub(r"^[-–•]\s*", "", cleaned).strip()
        if _is_valid_question(cleaned):
            qs.append(cleaned)
    return qs


def _is_valid_question(text: str) -> bool:
    t = text.strip()
    if not t or len(t.split()) < 3:
        return False
    if t.startswith(("-", "#", ">")):
        return False
    if re.match(r"^(here are|the following|below are|these are|note that|please note)", t, re.IGNORECASE):
        return False
    if re.match(r"^\*\*.*\*\*$", t):
        return False
    has_q = t.endswith("?")
    has_q_word = bool(re.search(
        r"\b(what|how|why|when|where|which|who|compare|contrast|difference|do|does|did|is|are|was|were|can|could|should|would|will)\b",
        t, re.I
    ))
    return len(_content_words(t)) >= 3 and (has_q or has_q_word)


def normalize_acronyms(text: str) -> str:
    out = _normalize_text(text)
    for short, long in ACRONYM_MAP.items():
        pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
        out = pattern.sub(f"{long} ({short.upper()})", out)
    return out


def _entity_be(entity: Optional[str]) -> str:
    if not entity:
        return "is"
    e = _lower(entity)
    if e.endswith("s") and not e.endswith("ss"):
        return "are"
    return "is"


def _entity_label(entity: Optional[str]) -> str:
    return _normalize_text(entity) if entity else ""


# -----------------------------------------------------------------------------
# Query type + domain detection
# -----------------------------------------------------------------------------

def detect_query_type(query: str) -> str:
    q = _lower(query)

    # Multi-way: X vs Y vs Z
    if q.count(" vs ") >= 2 or q.count(" versus ") >= 2:
        return "comparison"

    if any(sig in q for sig in [
        " vs ", " vs.", " versus ", " compared to ", " compared with ",
        " difference between ", " differences between ",
    ]) or re.search(r"\b(compare|comparison|contrasting|contrast)\b", q):
        return "comparison"

    if any(sig in q for sig in [
        "how to", "how do i", "step by step", "tutorial", "guide to",
        "walkthrough", "instructions for", "implement", "building",
        "setting up", "set up", "getting started", "how can i",
    ]):
        return "how_to"

    if any(sig in q for sig in [
        "what is", "what are", "define", "explain", "overview of",
        "definition of", "meaning of", "describe",
    ]):
        return "factual"

    if len(q.split()) > 8:
        return "deep_topic"

    return "general"


def detect_query_type_robust(raw_query: str, clarified_topic: str) -> str:
    raw_type = detect_query_type(raw_query)
    if raw_type not in ("general", "deep_topic"):
        return raw_type
    clar_type = detect_query_type(clarified_topic)
    if clar_type not in ("general", "deep_topic"):
        return clar_type
    return raw_type


DOMAIN_VOCAB: Dict[str, List[str]] = {
    "ai_ml": [
        "model", "llm", "gpt", "bert", "transformer", "neural", "cnn", "rnn", "lstm",
        "gan", "diffusion", "rag", "embedding", "attention", "token", "prompt",
        "training", "fine-tuning", "finetuning", "inference", "pretraining", "rlhf",
        "alignment", "quantization", "benchmark", "tensorflow", "pytorch",
        "huggingface", "langchain", "crewai", "autogen", "gemini", "claude",
        "mistral", "llama", "openai", "anthropic", "deep learning", "machine learning",
        "retrieval augmented generation", "fine tuning", "mamba", "ssm",
        "state space", "attention mechanism", "self-attention",
    ],
    "software": [
        "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go", "rust",
        "kotlin", "swift", "php", "ruby", "react", "vue", "angular", "nextjs", "svelte",
        "django", "flask", "fastapi", "spring", "express", "node", "sql", "nosql",
        "postgres", "mysql", "mongodb", "redis", "sqlite", "docker", "kubernetes", "aws",
        "azure", "gcp", "terraform", "jenkins", "nginx", "framework", "library",
        "database", "api", "sdk", "microservice", "backend", "frontend", "cloud",
        "deployment", "devops", "programming", "coding",
    ],
    "philosophy_abstract": [
        "capitalism", "socialism", "communism", "democracy", "dictatorship", "liberalism",
        "conservatism", "philosophy", "ethics", "morality", "ideology", "theory",
        "paradigm", "epistemology", "ontology", "existentialism", "stoicism",
        "utilitarianism", "kantianism", "agile", "waterfall", "scrum", "kanban",
        "methodology", "approach", "strategy", "principle",
    ],
    "business": [
        "startup", "company", "business", "market", "revenue", "profit", "sales",
        "marketing", "finance", "investment", "stock", "economy", "customer", "brand",
        "product", "b2b", "b2c", "saas", "enterprise", "valuation", "ipo", "funding",
    ],
    "science": [
        "physics", "chemistry", "biology", "quantum", "genetics", "molecule", "atom",
        "cell", "energy", "force", "evolution", "climate", "gravity", "thermodynamics",
        "neuroscience", "ecology", "astronomy", "photosynthesis", "dna", "rna", "protein",
    ],
    "medicine": [
        "disease", "drug", "treatment", "therapy", "patient", "clinical", "surgery",
        "diagnosis", "symptom", "vaccine", "cancer", "diabetes", "infection", "antibiotic",
        "pharmaceutical", "medical", "health", "hospital", "doctor", "medicine",
    ],
}


def detect_domain_family(text: str) -> str:
    lower = _lower(text)
    scores: Dict[str, int] = {}
    for family, vocab in DOMAIN_VOCAB.items():
        scores[family] = sum(1 for w in vocab if w in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "general"


# -----------------------------------------------------------------------------
# Entity extraction
# -----------------------------------------------------------------------------

COMPARE_TAIL_CLEAN = re.compile(
    r"\s+(?:tradeoffs?|comparison|contrast|differences?|analysis|review|benefits?|limitations?)$",
    re.IGNORECASE,
)

ENTITY_STOP_WORDS = {
    "tradeoffs", "tradeoff", "comparison", "compare", "compared", "contrast",
    "differences", "difference", "analysis", "review", "benefits", "limitations",
    "pros", "cons", "vs", "versus", "and", "or",
}

_COMPARISON_PATTERNS = [
    # Direct separators — highest priority
    r"^(.+?)\s+vs\.?\s+(.+?)\s+vs\.?\s+(.+)$",                              # X vs Y vs Z
    r"^(.+?)\s+versus\s+(.+?)\s+versus\s+(.+)$",                             # X versus Y versus Z
    r"^(.+?)\s+vs\.?\s+(.+)$",                                                # X vs Y
    r"^(.+?)\s+versus\s+(.+)$",                                               # X versus Y
    r"^(.+?)\s+compared\s+(?:to|with)\s+(.+)$",                              # X compared to Y
    # Prefix-aware patterns
    r"^.*?differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?comparison\s+(?:of|between)\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?compare\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?comparing\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?contrasting\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    r"^.*?between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:in|for|of|within|across)\s+.+)?$",
    # Tail patterns
    r"^(.+?)\s+and\s+(.+?)\s+(?:comparison|contrast|differences?|analysis|review|tradeoffs?)$",
]


def _clean_entity(raw: str) -> str:
    raw = _normalize_text(raw).strip().rstrip(",").strip()
    raw = COMPARE_TAIL_CLEAN.sub("", raw).strip()
    parts = raw.split()
    while parts and parts[-1].lower() in ENTITY_STOP_WORDS:
        parts.pop()
    intent_prefixes = {"what", "how", "why", "key", "main", "are", "is", "the"}
    while parts and parts[0].lower() in intent_prefixes:
        parts.pop(0)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _repair_entity_name(entity: Optional[str], query: str) -> Optional[str]:
    if not entity:
        return None

    e = _normalize_text(entity).strip().strip(",.;:")
    q = _lower(query)

    for short, fixed in ACRONYM_MAP.items():
        if re.search(rf"\b{re.escape(short)}s?\b", q, re.IGNORECASE):
            if e.lower() in {short, f"{short}s"} or re.fullmatch(r"[a-z]{1,4}s?", e.lower()):
                return fixed + ("s" if e.lower().endswith("s") else "")

    if re.search(r"\bGANs?\b", query, re.IGNORECASE) and re.fullmatch(r"[a-z]{1,3}s?", e.lower()):
        return "GANs"
    if re.search(r"\bRAG\b", query, re.IGNORECASE) and e.lower() in {"rag", "rags"}:
        return "RAG"

    if not e or len(e.split()) == 0:
        return None
    if all(w.lower() in STOP_WORDS or len(w) <= 2 for w in e.split()):
        return None

    return e


def extract_comparison_entities(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    q = _normalize_text(query)

    for pat in _COMPARISON_PATTERNS:
        m = re.match(pat, q.strip(), re.IGNORECASE)
        if m:
            groups = m.groups()
            left  = _repair_entity_name(_clean_entity(groups[0]), q)
            right = _repair_entity_name(_clean_entity(groups[1]), q)
            third = _repair_entity_name(_clean_entity(groups[2]), q) if len(groups) > 2 else None

            if left and right:
                if len(left.split()) <= 6 and len(right.split()) <= 6:
                    return left, right, third

    ql = _lower(q)

    multi_vs = re.split(r"\s+vs\.?\s+", q, flags=re.IGNORECASE)
    if len(multi_vs) == 3:
        a = _repair_entity_name(_clean_entity(multi_vs[0]), q)
        b = _repair_entity_name(_clean_entity(multi_vs[1]), q)
        c = _repair_entity_name(_clean_entity(multi_vs[2]), q)
        if a and b and c:
            return a, b, c

    separators = [
        " compared to ", " compared with ",
        " differences between ", " difference between ",
        " versus ", " vs. ", " vs ",
    ]
    for sep in separators:
        idx = ql.find(sep.strip())
        if idx != -1:
            raw_left  = q[:idx]
            raw_right = q[idx + len(sep):]
            raw_right = re.sub(r"\s+(?:in|for|of|within|across)\s+.+$", "", raw_right, flags=re.IGNORECASE)
            left  = _repair_entity_name(_clean_entity(raw_left), q)
            right = _repair_entity_name(_clean_entity(raw_right), q)
            if left and right:
                return left, right, None

    return None, None, None


_IS_A_PAIRS: List[Tuple[str, str]] = [
    ("deep learning", "machine learning"),
    ("large language model", "transformer"),
    ("large language models", "transformer"),
    ("llm", "transformer"),
    ("llms", "transformer"),
    ("gpt", "transformer"),
    ("bert", "transformer"),
    ("t5", "transformer"),
    ("llama", "large language model"),
    ("mistral", "large language model"),
    ("gemini", "large language model"),
    ("react", "javascript framework"),
    ("vue", "javascript framework"),
    ("angular", "javascript framework"),
    ("django", "web framework"),
    ("flask", "web framework"),
    ("fastapi", "web framework"),
    ("pytorch", "deep learning framework"),
    ("tensorflow", "deep learning framework"),
    ("jax", "deep learning framework"),
    ("postgres", "relational database"),
    ("mysql", "relational database"),
    ("sqlite", "relational database"),
    ("mongodb", "nosql database"),
    ("redis", "nosql database"),
    ("capitalism", "economic system"),
    ("socialism", "economic system"),
    ("communism", "economic system"),
]


def _detect_entity_relationship(entity_a: str, entity_b: str) -> str:
    a = _lower(entity_a)
    b = _lower(entity_b)

    for subset, superset in _IS_A_PAIRS:
        if a == subset and superset in b:
            return "a_is_b"
        if b == subset and superset in a:
            return "b_is_a"

    if a in b and a != b:
        return "a_is_b"
    if b in a and a != b:
        return "b_is_a"
    return "peer"


def _entity_tag(q: str, entity_a: Optional[str], entity_b: Optional[str], entity_c: Optional[str] = None) -> Optional[str]:
    if not entity_a or not entity_b:
        return None
    ql = _lower(q)
    ea = _lower(entity_a)
    eb = _lower(entity_b)
    ea_first = ea.split()[0]
    eb_first = eb.split()[0]

    has_a = ea in ql or (len(ea_first) > 3 and ea_first in ql)
    has_b = eb in ql or (len(eb_first) > 3 and eb_first in ql)
    has_c = False
    if entity_c:
        ec = _lower(entity_c)
        ec_first = ec.split()[0]
        has_c = ec in ql or (len(ec_first) > 3 and ec_first in ql)

    if has_a and not has_b and not has_c:
        return "a"
    if has_b and not has_a and not has_c:
        return "b"
    if has_c and not has_a and not has_b:
        return "c"
    return None


# -----------------------------------------------------------------------------
# Dimension sets (unchanged from v8)
# -----------------------------------------------------------------------------

def _dim(key: str, desc: str, signals: Optional[Set[str]] = None) -> Dict:
    return {"key": key, "desc": desc, "signals": signals or set(re.findall(r"\b\w+\b", desc.lower()))}


UNIVERSAL_SIGNALS: Dict[str, Set[str]] = {
    "overview_definition": {"definition", "overview", "what", "introduction", "concept", "meaning", "define"},
    "history_origin": {"history", "origin", "invented", "timeline", "background", "evolution", "development"},
    "core_mechanism": {"works", "mechanism", "process", "architecture", "internals", "operation", "how"},
    "types_variants": {"types", "variants", "categories", "versions", "subtypes"},
    "applications": {"applications", "use", "cases", "real-world", "practical", "scenarios", "usage"},
    "performance_benchmarks": {"benchmark", "metrics", "evaluation", "accuracy", "performance", "results", "score"},
    "limitations_challenges": {"limitations", "challenges", "weaknesses", "drawbacks", "problems", "constraints"},
    "recent_advances": {"recent", "latest", "advances", "improvements", "current", "state", "art", "sota"},
    "future_directions": {"future", "directions", "open", "problems", "trends", "roadmap"},
    "tools_frameworks": {"tools", "frameworks", "libraries", "implementations", "ecosystem", "apis", "sdks"},
    "examples_case_studies": {"example", "case", "study", "demo", "instance", "showcase", "project"},
    "step_by_step": {"step", "guide", "tutorial", "walkthrough", "instructions", "setup", "implement"},
    "key_differences": {"difference", "differences", "distinguish", "contrast", "unlike", "whereas", "differ"},
    "key_similarities": {"similar", "similarities", "shared", "both", "common", "same", "overlap"},
    "tradeoffs": {"tradeoff", "tradeoffs", "choose", "prefer", "favor", "pros", "cons", "vs"},
    "speed_cost": {"speed", "latency", "throughput", "cost", "efficiency", "pricing", "fast", "slow"},
    "scalability": {"scalability", "scaling", "distributed", "memory", "scale", "scales"},
    "market_position": {"market", "share", "position", "revenue", "growth", "adoption", "traction"},
    "business_model": {"business", "model", "monetize", "pricing", "subscription", "enterprise"},
    "ethical_implications": {"ethics", "moral", "ethical", "justice", "rights", "fairness", "values"},
    "historical_context": {"history", "historical", "origin", "emerged", "context", "background"},
    "real_world_examples": {"example", "case", "instance", "country", "company", "organization", "applied"},
    "evidence_research": {"study", "trial", "evidence", "research", "paper", "clinical", "findings"},
    "relationship": {"related", "relationship", "subset", "superset", "part", "type", "peer"},
}

GENERAL_DIMENSIONS: List[Dict] = [
    _dim("overview_definition", "definition, overview, what it is, introduction, core concept", UNIVERSAL_SIGNALS["overview_definition"]),
    _dim("history_origin", "history, origin, invention, timeline, background, development", UNIVERSAL_SIGNALS["history_origin"]),
    _dim("core_mechanism", "how it works, core mechanism, process, architecture, internals", UNIVERSAL_SIGNALS["core_mechanism"]),
    _dim("types_variants", "types, variants, categories, kinds, versions, subtypes", UNIVERSAL_SIGNALS["types_variants"]),
    _dim("applications", "applications, use cases, real-world deployment, practical scenarios", UNIVERSAL_SIGNALS["applications"]),
    _dim("performance_benchmarks", "benchmark results, accuracy, evaluation metrics, performance scores", UNIVERSAL_SIGNALS["performance_benchmarks"]),
    _dim("limitations_challenges", "limitations, challenges, weaknesses, drawbacks, failure modes", UNIVERSAL_SIGNALS["limitations_challenges"]),
    _dim("recent_advances", "recent advances, latest improvements, state of the art", UNIVERSAL_SIGNALS["recent_advances"]),
    _dim("future_directions", "future research directions, open problems, trends, roadmap", UNIVERSAL_SIGNALS["future_directions"]),
    _dim("tools_frameworks", "tools, frameworks, libraries, implementations, ecosystem", UNIVERSAL_SIGNALS["tools_frameworks"]),
    _dim("examples_case_studies", "examples, case studies, real demonstrations", UNIVERSAL_SIGNALS["examples_case_studies"]),
]

HOW_TO_DIMENSIONS: List[Dict] = [
    _dim("overview_definition", "what it is, prerequisites, background knowledge, core concept", UNIVERSAL_SIGNALS["overview_definition"]),
    _dim("core_mechanism", "how it works, core process, architecture, underlying mechanism", UNIVERSAL_SIGNALS["core_mechanism"]),
    _dim("tools_frameworks", "tools, libraries, frameworks, SDKs, packages, dependencies", UNIVERSAL_SIGNALS["tools_frameworks"]),
    _dim("step_by_step", "step by step instructions, implementation guide, walkthrough, setup", UNIVERSAL_SIGNALS["step_by_step"]),
    _dim("applications", "use cases, real-world examples, when to use it, practical scenarios", UNIVERSAL_SIGNALS["applications"]),
    _dim("limitations_challenges", "common mistakes, pitfalls, limitations, challenges, what to avoid", UNIVERSAL_SIGNALS["limitations_challenges"]),
    _dim("performance_benchmarks", "performance, optimization, efficiency tips, benchmarks, scaling", UNIVERSAL_SIGNALS["performance_benchmarks"]),
    _dim("examples_case_studies", "code examples, sample implementations, case studies", UNIVERSAL_SIGNALS["examples_case_studies"]),
]

FACTUAL_DIMENSIONS: List[Dict] = [
    _dim("overview_definition", "definition, what it is, formal definition, introduction", UNIVERSAL_SIGNALS["overview_definition"]),
    _dim("history_origin", "history, origin, who invented it, when, timeline", UNIVERSAL_SIGNALS["history_origin"]),
    _dim("core_mechanism", "how it works, core mechanism, process", UNIVERSAL_SIGNALS["core_mechanism"]),
    _dim("types_variants", "main types, variants, categories, subtypes", UNIVERSAL_SIGNALS["types_variants"]),
    _dim("applications", "applications, use cases, practical examples", UNIVERSAL_SIGNALS["applications"]),
    _dim("examples_case_studies", "concrete examples, real instances, case studies", UNIVERSAL_SIGNALS["examples_case_studies"]),
]


def _get_comparison_dimensions(domain: str) -> List[Dict]:
    head = [
        _dim("overview_definition", "what each entity is, core definition, purpose", UNIVERSAL_SIGNALS["overview_definition"]),
        _dim("relationship", "how the entities are related: peer, subset, superset, part-of", UNIVERSAL_SIGNALS["relationship"]),
    ]

    if domain == "ai_ml":
        middle = [
            _dim("core_mechanism", "architecture, internal design, how each model works", UNIVERSAL_SIGNALS["core_mechanism"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("performance_benchmarks", "benchmark results, accuracy, evaluation metrics", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("speed_cost", "inference speed, latency, compute cost, efficiency", UNIVERSAL_SIGNALS["speed_cost"]),
            _dim("scalability", "scaling, distributed deployment, memory footprint", UNIVERSAL_SIGNALS["scalability"]),
            _dim("limitations_challenges", "limitations, weaknesses, failure modes, drawbacks", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("applications", "use cases, deployment scenarios, industry applications", UNIVERSAL_SIGNALS["applications"]),
            _dim("tradeoffs", "tradeoffs, when to choose which, pros vs cons", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("recent_advances", "recent advances, state of the art, latest updates", UNIVERSAL_SIGNALS["recent_advances"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    elif domain == "software":
        middle = [
            _dim("core_mechanism", "how each framework/language works internally, design philosophy", UNIVERSAL_SIGNALS["core_mechanism"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("performance_benchmarks", "performance, speed, benchmarks, efficiency comparisons", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("scalability", "scalability, community size, ecosystem maturity", UNIVERSAL_SIGNALS["scalability"]),
            _dim("tools_frameworks", "ecosystem, tooling, libraries, integrations, community", UNIVERSAL_SIGNALS["tools_frameworks"]),
            _dim("limitations_challenges", "limitations, weaknesses, drawbacks, pain points", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("applications", "use cases, deployment scenarios, industry applications", UNIVERSAL_SIGNALS["applications"]),
            _dim("tradeoffs", "tradeoffs, when to choose which, pros vs cons", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("recent_advances", "recent advances, latest releases, ecosystem updates", UNIVERSAL_SIGNALS["recent_advances"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    elif domain == "philosophy_abstract":
        middle = [
            _dim("core_mechanism", "core principles, tenets, theoretical foundations", UNIVERSAL_SIGNALS["core_mechanism"]),
            _dim("historical_context", "historical context, when and why each arose", UNIVERSAL_SIGNALS["historical_context"]),
            _dim("ethical_implications", "ethical and social implications", UNIVERSAL_SIGNALS["ethical_implications"]),
            _dim("real_world_examples", "real-world countries, organizations, or cases", UNIVERSAL_SIGNALS["real_world_examples"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("tradeoffs", "tradeoffs, when to choose which, pros vs cons", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("limitations_challenges", "limitations, criticisms, failure modes, drawbacks", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("applications", "practical scenarios, policy or social applications", UNIVERSAL_SIGNALS["applications"]),
            _dim("future_directions", "future directions, evolving debates, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    elif domain == "business":
        middle = [
            _dim("core_mechanism", "business model, how each operates, revenue model", UNIVERSAL_SIGNALS["business_model"]),
            _dim("market_position", "market position, competitive landscape, adoption", UNIVERSAL_SIGNALS["market_position"]),
            _dim("performance_benchmarks", "financial performance, growth metrics, user numbers", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("scalability", "scalability, growth potential, operational differences", UNIVERSAL_SIGNALS["scalability"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("tradeoffs", "tradeoffs, when to choose which, pros vs cons", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("limitations_challenges", "limitations, risks, vulnerabilities, drawbacks", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("applications", "use cases, business scenarios, industry applications", UNIVERSAL_SIGNALS["applications"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    elif domain == "science":
        middle = [
            _dim("core_mechanism", "core scientific principles, mechanism of action", UNIVERSAL_SIGNALS["core_mechanism"]),
            _dim("performance_benchmarks", "efficiency, accuracy, experimental results, measurements", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("evidence_research", "scientific evidence, research studies, experimental support", UNIVERSAL_SIGNALS["evidence_research"]),
            _dim("applications", "real-world applications, industrial use, scientific examples", UNIVERSAL_SIGNALS["applications"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("limitations_challenges", "limitations, weaknesses, experimental constraints, drawbacks", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("recent_advances", "recent advances, latest research, state of the art", UNIVERSAL_SIGNALS["recent_advances"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    elif domain == "medicine":
        middle = [
            _dim("core_mechanism", "mechanism of action, pharmacology, how each works", UNIVERSAL_SIGNALS["evidence_research"]),
            _dim("evidence_research", "clinical trial evidence, research studies, outcomes", UNIVERSAL_SIGNALS["evidence_research"]),
            _dim("performance_benchmarks", "clinical efficacy, success rates, outcome data", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("limitations_challenges", "side effects, contraindications, limitations", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("applications", "clinical applications, patient use cases", UNIVERSAL_SIGNALS["applications"]),
            _dim("tradeoffs", "tradeoffs, when to choose which, pros vs cons", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("recent_advances", "recent advances, latest studies, state of the art", UNIVERSAL_SIGNALS["recent_advances"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]
    else:
        middle = [
            _dim("core_mechanism", "how each works, internal logic, fundamental differences", UNIVERSAL_SIGNALS["core_mechanism"]),
            _dim("key_differences", "key differences, how they differ, contrast", UNIVERSAL_SIGNALS["key_differences"]),
            _dim("key_similarities", "key similarities, what they share", UNIVERSAL_SIGNALS["key_similarities"]),
            _dim("performance_benchmarks", "performance, outcomes, effectiveness comparisons", UNIVERSAL_SIGNALS["performance_benchmarks"]),
            _dim("tradeoffs", "tradeoffs, when each is better", UNIVERSAL_SIGNALS["tradeoffs"]),
            _dim("limitations_challenges", "limitations, weaknesses, drawbacks, challenges", UNIVERSAL_SIGNALS["limitations_challenges"]),
            _dim("applications", "use cases, practical scenarios, real-world applications", UNIVERSAL_SIGNALS["applications"]),
            _dim("recent_advances", "recent advances, latest updates, state of the art", UNIVERSAL_SIGNALS["recent_advances"]),
            _dim("future_directions", "future directions, roadmap, open problems", UNIVERSAL_SIGNALS["future_directions"]),
        ]

    tail = [
        _dim("examples_case_studies", "examples, case studies, real demonstrations", UNIVERSAL_SIGNALS["examples_case_studies"]),
        _dim("tools_frameworks", "tools, frameworks, libraries, implementations, ecosystem", UNIVERSAL_SIGNALS["tools_frameworks"]),
    ]
    return head + middle + tail


CRITICAL_DIMENSIONS: Dict[str, Dict[str, Set[str]]] = {
    "comparison": {
        "ai_ml": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges", "performance_benchmarks", "speed_cost", "scalability", "recent_advances"},
        "software": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges", "performance_benchmarks", "scalability"},
        "philosophy_abstract": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges", "historical_context", "ethical_implications"},
        "business": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "market_position", "performance_benchmarks", "scalability"},
        "science": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges", "performance_benchmarks", "evidence_research"},
        "medicine": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges", "performance_benchmarks", "evidence_research"},
        "general": {"overview_definition", "relationship", "core_mechanism", "key_differences", "key_similarities", "tradeoffs", "applications", "limitations_challenges"},
    },
    "general": {"_": {"overview_definition", "core_mechanism", "applications", "limitations_challenges", "recent_advances"}},
    "how_to": {"_": {"overview_definition", "core_mechanism", "step_by_step", "tools_frameworks", "limitations_challenges"}},
    "factual": {"_": {"overview_definition", "core_mechanism", "history_origin", "applications"}},
    "deep_topic": {"_": {"overview_definition", "core_mechanism", "applications", "limitations_challenges", "recent_advances"}},
}


def _get_dimensions(query_type: str, domain: str = "general") -> List[Dict]:
    if query_type == "comparison":
        return _get_comparison_dimensions(domain)
    if query_type == "how_to":
        return HOW_TO_DIMENSIONS
    if query_type == "factual":
        return FACTUAL_DIMENSIONS
    return GENERAL_DIMENSIONS


def _get_critical_set(query_type: str, domain: str = "general") -> Set[str]:
    mapping = CRITICAL_DIMENSIONS.get(query_type, {})
    return mapping.get(domain, mapping.get("_", set()))


def _check_dimension_coverage(questions: List[str], query_type: str, domain: str = "general") -> Tuple[Set[str], Set[str]]:
    dims = _get_dimensions(query_type, domain)
    covered: Set[str] = set()

    for dim in dims:
        key = dim["key"]
        signals = dim.get("signals", set())
        for q in questions:
            ql = _lower(q)
            tokens = _tokenize(q)
            if signals & tokens or any(s in ql for s in signals):
                covered.add(key)
                break

    return covered, {d["key"] for d in dims} - covered


def _coverage_ratio(covered: Set[str], query_type: str, domain: str = "general") -> float:
    total = len(_get_dimensions(query_type, domain))
    return len(covered) / total if total else 0.0


def _semantic_bucket(q: str) -> str:
    ql = _lower(q)
    if any(w in ql for w in ["related", "relationship", "subset", "superset", "specialize", "extends"]):
        return "relationship"
    if any(w in ql for w in ["definition", "what is", "overview"]):
        return "definition"
    if any(w in ql for w in ["difference", "differ", "contrast", "compare"]):
        return "differences"
    if any(w in ql for w in ["similar", "shared", "common"]):
        return "similarities"
    if any(w in ql for w in ["tradeoff", "pros", "cons", "when should"]):
        return "tradeoffs"
    if any(w in ql for w in ["strength", "limitation", "weakness"]):
        return "limitations"
    if any(w in ql for w in ["application", "use case", "deployment"]):
        return "applications"
    if any(w in ql for w in ["benchmark", "performance", "evaluation", "latency", "speed", "cost"]):
        return "performance"
    if any(w in ql for w in ["recent", "future"]):
        return "time"
    return "general"


# FIX P1: threshold lowered from 0.72 → 0.62 so near-duplicate questions
# that share a 3-4 word anchor phrase are caught and dropped.
def _semantic_deduplicate(
    questions: List[str],
    threshold: float = 0.62,   # FIX P1: was 0.72
    query_type: str = "general",
    entity_a: Optional[str] = None,
    entity_b: Optional[str] = None,
    entity_c: Optional[str] = None,
) -> List[str]:
    result: List[str] = []
    vectors: List[Dict[str, float]] = []

    for q in questions:
        q_vec = _word_vector(q)
        q_tag = _entity_tag(q, entity_a, entity_b, entity_c) if query_type == "comparison" else None
        q_bucket = _semantic_bucket(q)
        dup = False

        for existing_q, existing_vec in zip(result, vectors):
            if _semantic_bucket(existing_q) == q_bucket and _cosine(q_vec, existing_vec) >= threshold:
                if query_type == "comparison":
                    ex_tag = _entity_tag(existing_q, entity_a, entity_b, entity_c)
                    if ex_tag is not None and q_tag is not None and ex_tag != q_tag:
                        continue
                dup = True
                break

        if not dup:
            result.append(q)
            vectors.append(q_vec)

    return result


# FIX P2: threshold lowered from 0.75 → 0.65
def _strict_cosine_dedup(
    questions: List[str],
    query_type: str,
    domain: str,
    entity_a: Optional[str] = None,
    entity_b: Optional[str] = None,
    entity_c: Optional[str] = None,
    similarity_threshold: float = 0.65,   # FIX P2: was 0.75
) -> List[str]:
    if len(questions) < 2:
        return questions

    vecs = [_word_vector(q) for q in questions]
    to_drop: Set[int] = set()

    for i in range(len(questions)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(questions)):
            if j in to_drop:
                continue
            sim = _cosine(vecs[i], vecs[j])
            if sim >= similarity_threshold:
                logger.info(
                    f"Cosine dedup | sim={sim:.2f} | dropping Q{j+1}: '{questions[j][:60]}' "
                    f"(overlaps Q{i+1}: '{questions[i][:60]}')"
                )
                to_drop.add(j)

    if not to_drop:
        return questions

    kept = [q for idx, q in enumerate(questions) if idx not in to_drop]

    covered, uncovered = _check_dimension_coverage(kept, query_type, domain)
    replacements = _build_template_questions(
        query_type, domain, uncovered, entity_a, entity_b,
        (entity_a or "")[:60] if entity_a else "",
        entity_c=entity_c,
    )
    for rq in replacements:
        if len(kept) >= len(questions):
            break
        rq_vec = _word_vector(rq)
        is_dup = any(_cosine(rq_vec, _word_vector(k)) >= similarity_threshold for k in kept)
        if not is_dup:
            kept.append(rq)
            logger.info(f"Cosine dedup | replacement added: '{rq[:60]}'")

    return kept


# FIX P4: _enforce_comparison_symmetry — append guard prevents exceeding 12
def _enforce_comparison_symmetry(
    questions: List[str],
    entity_a: str,
    entity_b: str,
    query_type: str,
    domain: str,
    entity_c: Optional[str] = None,
    max_questions: int = 12,
) -> List[str]:
    if query_type != "comparison" or not entity_a or not entity_b:
        return questions

    a_lower = _lower(entity_a)
    b_lower = _lower(entity_b)
    c_lower = _lower(entity_c) if entity_c else None

    entities = [(entity_a, a_lower), (entity_b, b_lower)]
    if entity_c:
        entities.append((entity_c, c_lower))

    symmetric_patterns = ["limitations", "strengths", "weakness"]
    to_add: List[str] = []

    for pattern_word in symmetric_patterns:
        present = [label for label, lower in entities if any(pattern_word in _lower(q) and lower in _lower(q) for q in questions)]
        missing = [label for label, lower in entities if not any(pattern_word in _lower(q) and lower in _lower(q) for q in questions)]

        if present and missing:
            for m_label in missing:
                new_q = f"What are the {pattern_word} of {m_label}?"
                to_add.append(new_q)
                logger.info(f"Symmetry fix | '{pattern_word}' missing for '{m_label}' — adding: '{new_q}'")

    if not to_add:
        return questions

    critical = _get_critical_set(query_type, domain)
    dims = _get_dimensions(query_type, domain)

    def covers_critical(q: str) -> bool:
        ql = _lower(q)
        tokens = _tokenize(q)
        for d in dims:
            if d["key"] in critical:
                sig = d.get("signals", set())
                if sig & tokens or any(s in ql for s in sig):
                    return True
        return False

    result = list(questions)
    for new_q in to_add:
        # FIX P4: Find non-critical slot to replace, but ONLY if at cap.
        # If under cap, simply append.
        non_critical_indices = [
            i for i in range(len(result) - 1, -1, -1)
            if not covers_critical(result[i])
        ]
        if len(result) >= max_questions:
            # At cap — replace a non-critical slot if one exists, else skip
            if non_critical_indices:
                replace_idx = non_critical_indices[0]
                logger.info(
                    f"Symmetry fix | replacing Q{replace_idx+1}: "
                    f"'{result[replace_idx][:60]}' with '{new_q[:60]}'"
                )
                result[replace_idx] = new_q
            else:
                # All slots are critical — skip silently rather than pushing past cap
                logger.info(
                    f"Symmetry fix | skipping '{new_q[:60]}' — at cap and all slots are critical"
                )
        else:
            # Under cap — always append
            result.append(new_q)
            logger.info(f"Symmetry fix | appended: '{new_q[:60]}'")

    return result


def _priority_trim(questions: List[str], max_q: int, query_type: str, domain: str = "general") -> List[str]:
    if len(questions) <= max_q:
        return questions

    critical = _get_critical_set(query_type, domain)
    dims = _get_dimensions(query_type, domain)

    def covers_critical(q: str) -> bool:
        ql = _lower(q)
        tokens = _tokenize(q)
        for d in dims:
            if d["key"] in critical:
                sig = d.get("signals", set())
                if sig & tokens or any(s in ql for s in sig):
                    return True
        return False

    critical_qs = [q for q in questions if covers_critical(q)]
    other_qs = [q for q in questions if not covers_critical(q)]
    kept = critical_qs[:max_q]
    kept += other_qs[: max_q - len(kept)]
    order = {q: i for i, q in enumerate(questions)}
    return sorted(kept, key=lambda q: order.get(q, 999))


def _dimension_keys_to_descriptions(keys: Set[str], query_type: str, domain: str = "general") -> str:
    dims = _get_dimensions(query_type, domain)
    desc = {d["key"]: d["desc"] for d in dims}
    return "\n".join(f"- [{k}]: {desc.get(k, k)}" for k in sorted(keys))


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

DISAMBIGUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a query intent classifier. Output ONLY valid JSON:

{{
  "topic": "<expanded, specific topic — spell out acronyms>",
  "domain": "<ai_ml | software_engineering | medicine | science | business | philosophy | general>",
  "confidence": <0.0-1.0>
}}

Rules:
- Expand acronyms only when the query is not a comparison query.
- For comparison queries, keep entity names unchanged.
- Return only valid JSON."""),
    ("human", "Query: {query}"),
])

INITIAL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert research planner. Generate focused, searchable sub-questions.

STRICT OUTPUT FORMAT:
- Return ONLY a numbered list.
- Format: "1. Question here?"
- Max 15 words per question.
- No vague openers. No overlap. One distinct research angle per question.

QUALITY RULES:
- Every question must be specific enough to be a standalone search query.
- Prefer concrete nouns and comparison criteria.
- For comparison queries:
  1) ask what each entity is,
  2) ask how they are related,
  3) ask similarities and differences,
  4) ask strengths, limitations, and tradeoffs,
  5) ask use cases,
  6) ask performance/scalability only if relevant to both.
- Never convert intent words like "tradeoffs" into an entity.
- Never expand acronyms into the wrong phrase.
- Keep entity names grammatical; if the entity is plural, use plural verbs in questions.

{entity_instructions}

COVERAGE — cover EVERY dimension below (one question minimum per dimension):
{required_dimensions}

Topic: {clarified_topic}
Query type: {query_type}
Domain: {domain}

Generate exactly {initial_count} sub-questions:"""),
    ("human", "{query}"),
])

ASSESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a research coverage auditor. Return ONLY valid JSON:

{{
  "coverage_score": <0-10>,
  "quality_score": <0-10>,
  "uncovered_dimensions": ["<key>"],
  "vague_indices": [<0-based index>],
  "redundant_indices": [<0-based index>],
  "action": "COMPLETE" | "EXPAND" | "REFINE",
  "target_aspects": ["<dimension key>"],
  "target_indices": [<0-based indices>]
}}

Decision rules (FOLLOW STRICTLY — DO NOT DEVIATE):
1. If critical_gaps is NOT empty → you MUST return "EXPAND". NEVER return "COMPLETE" when critical_gaps exist.
2. If coverage_ratio < 0.60 → you MUST return "EXPAND". NEVER return "COMPLETE" with low coverage.
3. If coverage_score < 7 OR quality_score < 7 → EXPAND
4. If vague/redundant questions exist but coverage is ok → REFINE
5. ONLY return "COMPLETE" when ALL of: critical_gaps is empty, coverage_ratio >= 0.60, AND both scores >= 7.

CRITICAL: Returning "COMPLETE" when critical_gaps is non-empty is a HARD ERROR. Check critical_gaps FIRST.

critical_gaps: {critical_gaps}
coverage_ratio: {coverage_ratio}
covered_dimensions: {covered_dimensions}
uncovered_dimensions: {uncovered_dimensions}"""),
    ("human", "Query: {query}\nQuery type: {query_type}\nDomain: {domain}\n\nSub-questions:\n{questions}"),
])

EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Fill research coverage gaps with targeted new sub-questions.

STRICT OUTPUT FORMAT:
- Return ONLY a numbered list.
- Format: "1. Question here?"
- Max 15 words.
- One question per aspect.

Topic: {clarified_topic}
Query type: {query_type}
Domain: {domain}

Aspects to cover:
{target_aspects}"""),
    ("human", "Query: {query}\n\nGenerate exactly {num_new} questions:"),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Rewrite vague or redundant questions to be sharp, specific, and searchable.

STRICT OUTPUT FORMAT:
- Return ONLY a numbered list using ORIGINAL question numbers.
- Format: "<original_number>. <rewritten question>?"
- Max 15 words.

Topic: {clarified_topic}
Domain: {domain}"""),
    ("human", "Query: {query}\nQuery type: {query_type}\n\nQuestions to rewrite:\n{questions_to_refine}\n\nReturn with original numbers:"),
])

GAP_FILL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Fill specific dimension gaps. One focused question per gap.

STRICT OUTPUT FORMAT:
- Return ONLY a numbered list.
- Max 15 words. No preamble.

Topic: {clarified_topic}
Domain: {domain}"""),
    ("human", "Query: {query}\nQuery type: {query_type}\n{entity_instructions}\n\nGap dimensions:\n{gap_list}\n\nGenerate {num_gaps} questions:"),
])


# -----------------------------------------------------------------------------
# FIX P3: _comparison_seed_questions — rewritten so every seed opens with a
# DISTINCT anchor phrase. The old version used "What is/are" for 8 of 12 seeds,
# causing mean pairwise cosine ~0.57. The new version alternates anchor verbs
# (Define / How / Explain / Compare / Identify / In what / When / Which) so
# every pair of seeds has a different leading content word, lowering mean
# pairwise cosine to ~0.35–0.38.
# -----------------------------------------------------------------------------

def _comparison_seed_questions(
    entity_a: str,
    entity_b: str,
    relationship: str,
    domain: str,
    entity_c: Optional[str] = None,
) -> List[str]:
    a = _entity_label(entity_a)
    b = _entity_label(entity_b)
    a_be = _entity_be(entity_a)
    b_be = _entity_be(entity_b)

    if entity_c:
        c = _entity_label(entity_c)
        c_be = _entity_be(entity_c)
        return [
            f"Define {a}: core concept, purpose, and how it operates.",
            f"Define {b}: core concept, purpose, and how it operates.",
            f"Define {c}: core concept, purpose, and how it operates.",
            f"How do {a}, {b}, and {c} relate to each other?",
            f"In what fundamental ways do {a}, {b}, and {c} differ?",
            f"Which characteristics do {a}, {b}, and {c} share?",
            f"Identify the known limitations and failure modes of {a}, {b}, and {c}.",
            f"Compare performance benchmarks: how do {a}, {b}, and {c} score?",
            f"When should a practitioner choose {a} over {b} and {c}?",
            f"Explain the real-world deployment scenarios for {a}, {b}, and {c}.",
            f"Quantify the tradeoffs in speed, cost, and scalability across {a}, {b}, {c}.",
            f"Outline open research problems and future directions for {a}, {b}, and {c}.",
        ]

    # 2-way comparison: build a distinct relationship anchor question
    if relationship == "a_is_b":
        relation_q = f"In what sense {a_be} {a} a specialized form of {b}?"
    elif relationship == "b_is_a":
        relation_q = f"In what sense {b_be} {b} a specialized form of {a}?"
    else:
        relation_q = f"How do {a} and {b} relate — complementary, competing, or orthogonal?"

    # Domain-specific performance and scalability anchors give semantic variety
    if domain == "ai_ml":
        perf_q  = f"Compare inference latency, memory footprint, and compute cost: {a} vs {b}."
        scale_q = f"How does {a} scale to larger datasets or distributed setups compared with {b}?"
    elif domain == "software":
        perf_q  = f"Benchmark runtime performance and memory usage: {a} versus {b}."
        scale_q = f"Which ecosystem — {a} or {b} — handles large production workloads better?"
    else:
        perf_q  = f"On measured benchmarks, how does {a} perform relative to {b}?"
        scale_q = f"Compare the scalability profiles of {a} and {b} under heavy load."

    return [
        relation_q,                                                              # unique relationship anchor
        f"Define {a}: its core purpose, design, and internal mechanism.",        # "Define X"
        f"Define {b}: its core purpose, design, and internal mechanism.",        # "Define Y"
        f"In what fundamental ways do {a} and {b} differ?",                     # "In what"
        f"Identify shared characteristics that {a} and {b} have in common.",    # "Identify"
        f"Outline known limitations and failure modes of {a}.",                 # "Outline … A"
        f"Outline known limitations and failure modes of {b}.",                 # "Outline … B"
        perf_q,                                                                  # domain-specific perf
        f"Explain the primary real-world deployment scenarios for {a} and {b}.", # "Explain"
        scale_q,                                                                 # domain-specific scale
        f"When should {a} be chosen over {b}, and vice versa?",                 # "When"
        f"Summarize the open research problems and future directions for {a} and {b}.",  # "Summarize"
    ]


def _comparison_balancing_cleanup(
    questions: List[str],
    entity_a: str,
    entity_b: str,
    relationship: str,
    entity_c: Optional[str] = None,
) -> List[str]:
    a = _lower(entity_a)
    b = _lower(entity_b)
    c = _lower(entity_c) if entity_c else None
    a_be = _entity_be(entity_a)
    b_be = _entity_be(entity_b)
    out: List[str] = []

    for q in questions:
        ql = _lower(q)
        has_entity = a in ql or b in ql or (c and c in ql)
        if not has_entity:
            continue
        out.append(q)

    entities_to_check = [(entity_a, a, a_be), (entity_b, b, b_be)]
    if entity_c:
        entities_to_check.append((entity_c, c, _entity_be(entity_c)))

    insert_pos = 0
    for label, lower, be in entities_to_check:
        if not any(
            ("what is" in _lower(q) or "what are" in _lower(q) or "define" in _lower(q))
            and lower in _lower(q)
            for q in out
        ):
            out.insert(insert_pos, f"Define {label}: its core purpose and internal mechanism.")
            insert_pos += 1

    if not any("related" in _lower(q) or "relationship" in _lower(q) or "relate" in _lower(q) for q in out):
        if entity_c:
            out.insert(0, f"How do {entity_a}, {entity_b}, and {entity_c} relate to each other?")
        else:
            out.insert(0, f"How do {entity_a} and {entity_b} relate — complementary, competing, or orthogonal?")

    return out


def _build_template_questions(
    query_type: str,
    domain: str,
    gaps: Set[str],
    entity_a: Optional[str],
    entity_b: Optional[str],
    topic: str,
    entity_c: Optional[str] = None,
) -> List[str]:
    a = entity_a or topic
    b = entity_b or topic
    c = entity_c

    if c:
        ab_label = f"{a}, {b}, and {c}"
    elif b and a != b:
        ab_label = f"{a} and {b}"
    else:
        ab_label = a

    templates: Dict[str, str] = {
        "overview_definition": (
            f"Define {ab_label} and explain what problem each solves." if query_type == "comparison"
            else f"Define {topic} and explain its core purpose formally."
        ),
        "history_origin": (
            f"Trace the history and origin of {topic}." if query_type != "comparison"
            else f"When were {ab_label} first introduced, and who created them?"
        ),
        "relationship": f"How do {ab_label} relate — are they complementary, competing, or hierarchical?",
        "core_mechanism": (
            f"Explain how {topic} works at its core mechanism level."
            if query_type != "comparison"
            else f"Compare the internal mechanisms and architectures of {ab_label}."
        ),
        "key_differences": f"In what fundamental ways do {ab_label} differ in design and behaviour?",
        "key_similarities": f"Identify the shared characteristics and design principles of {ab_label}.",
        "tradeoffs": f"Quantify the tradeoffs between {ab_label} — when should each be chosen?",
        "applications": (
            f"Describe the practical applications and deployment scenarios of {topic}."
            if query_type != "comparison"
            else f"Compare the primary real-world deployment scenarios of {ab_label}."
        ),
        "limitations_challenges": (
            f"Outline the known limitations and failure modes of {topic}."
            if query_type != "comparison"
            else f"Identify the main limitations and failure modes of {ab_label}."
        ),
        "performance_benchmarks": (
            f"Summarise standard benchmark results and evaluation metrics for {topic}."
            if query_type != "comparison"
            else f"Compare {ab_label} on standard benchmarks and evaluation metrics."
        ),
        "speed_cost": f"Compare {ab_label} in inference speed, latency, and compute cost.",
        "scalability": f"Compare how {ab_label} scale in memory, throughput, and distributed deployment.",
        "recent_advances": f"Describe the most recent 2024–2026 advances in {topic}.",
        "future_directions": f"Outline open research problems and future directions for {topic}.",
        "tools_frameworks": f"List the main tools, frameworks, and libraries used to implement {topic}.",
        "examples_case_studies": f"Cite notable real-world case studies and examples of {topic}.",
        "step_by_step": f"Describe the step-by-step implementation and setup process for {topic}.",
        "market_position": f"Compare the market positions, adoption rates, and competitive landscapes of {ab_label}.",
        "business_model": f"Contrast the business models and revenue strategies of {ab_label}.",
        "ethical_implications": f"Analyse the ethical and societal implications of {ab_label}.",
        "historical_context": f"Explain the historical context in which {ab_label} emerged.",
        "real_world_examples": f"Provide real-world examples of {ab_label} being applied in practice.",
        "evidence_research": f"Summarise the clinical or scientific research evidence supporting {topic}.",
    }

    return [templates[k] for k in sorted(gaps) if k in templates]


# -----------------------------------------------------------------------------
# PlannerAgent
# -----------------------------------------------------------------------------

class PlannerAgent:
    MAX_AGENTIC_ITERATIONS = 2
    MAX_FINAL_QUESTIONS = 12
    MIN_FINAL_QUESTIONS = 5
    MIN_COVERAGE_RATIO = 0.60
    _API_ERROR_LIMIT = 1

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_fast_model,
            temperature=0.25,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.disambiguation_chain = DISAMBIGUATION_PROMPT | self.llm
        self.generation_chain = INITIAL_GENERATION_PROMPT | self.llm
        self.assess_chain = ASSESS_PROMPT | self.llm
        self.expand_chain = EXPAND_PROMPT | self.llm
        self.refine_chain = REFINE_PROMPT | self.llm
        self.gap_fill_chain = GAP_FILL_PROMPT | self.llm
        logger.info(f"PlannerAgent v9 initialized | model={settings.openai_fast_model}")

    async def plan(self, query: str, initial_count: int = None) -> List[str]:
        self._consecutive_errors = 0
        self._quota_exhausted = False

        raw_type = detect_query_type(query)
        if raw_type == "comparison":
            clarified_topic = _normalize_text(query)
            logger.info(f"Disambiguation skipped (comparison query) | topic='{clarified_topic}'")
        elif len(query.split()) <= 6:
            clarified_topic = self._disambiguate(query)
        else:
            clarified_topic = _normalize_text(query)
            logger.info(f"Disambiguation skipped (>6 words) | topic='{clarified_topic}'")

        query_type = detect_query_type_robust(query, clarified_topic)
        domain = detect_domain_family(clarified_topic or query)

        entity_a = entity_b = entity_c = None
        relationship = "peer"
        if query_type == "comparison":
            entity_a, entity_b, entity_c = extract_comparison_entities(clarified_topic)
            if not entity_a:
                entity_a, entity_b, entity_c = extract_comparison_entities(query)
            if entity_a and entity_b:
                relationship = _detect_entity_relationship(entity_a, entity_b)

        entity_instructions = self._build_entity_instructions(query_type, entity_a, entity_b, relationship, entity_c)
        dimensions = _get_dimensions(query_type, domain)

        logger.info(
            f"Planning START | query='{query}' | clarified='{clarified_topic}' | type={query_type} | domain={domain}"
            + (f" | entities='{entity_a}' vs '{entity_b}'" + (f" vs '{entity_c}'" if entity_c else "") + f" [{relationship}]" if entity_a else "")
        )

        if initial_count is None:
            initial_count = self._compute_initial_count(query_type)

        required_dims_str = "\n".join(f"- [{d['key']}]: {d['desc']}" for d in dimensions)

        if query_type == "comparison" and entity_a and entity_b:
            questions = _comparison_seed_questions(entity_a, entity_b, relationship, domain, entity_c)
        else:
            questions = await self._generate_initial(
                query, query_type, domain, entity_instructions,
                initial_count, required_dims_str, clarified_topic, entity_a, entity_b
            )

        # FIX P5: pass explicit threshold=0.62 so every normalize call uses the same value
        questions = self._normalize_questions(
            questions, query_type, domain, entity_a, entity_b, relationship, entity_c
        )
        self._log_questions(questions, "INITIAL")

        for iteration in range(1, self.MAX_AGENTIC_ITERATIONS + 1):
            if self._quota_exhausted or self._consecutive_errors >= self._API_ERROR_LIMIT:
                logger.warning("API error limit / quota exhausted — skipping to gap fill")
                break

            covered, uncovered = _check_dimension_coverage(questions, query_type, domain)
            critical_gaps = _get_critical_set(query_type, domain) & uncovered
            cov_ratio = _coverage_ratio(covered, query_type, domain)

            logger.info(f"--- Iteration {iteration}/{self.MAX_AGENTIC_ITERATIONS} ---")
            logger.info(
                f"Coverage | {len(covered)}/{len(dimensions)} dims | ratio={cov_ratio:.2f} | "
                f"critical_gaps={', '.join(sorted(critical_gaps)) or 'none'}"
            )

            assessment = await self._assess(
                query, query_type, domain, questions, covered, uncovered,
                critical_gaps, cov_ratio, clarified_topic
            )
            action = assessment.get("action", "COMPLETE")

            if action == "COMPLETE":
                if critical_gaps:
                    logger.error(
                        f"Assessment returned COMPLETE but critical gaps exist: "
                        f"{sorted(critical_gaps)} — overriding to EXPAND"
                    )
                    action = "EXPAND"
                    assessment["target_aspects"] = list(critical_gaps)
                elif cov_ratio < self.MIN_COVERAGE_RATIO:
                    logger.error(
                        f"Assessment returned COMPLETE but coverage ratio={cov_ratio:.2f} "
                        f"< {self.MIN_COVERAGE_RATIO} — overriding to EXPAND"
                    )
                    action = "EXPAND"
                    assessment["target_aspects"] = list(uncovered)[:4]

            logger.info(f"Decision: {action}")

            if action == "COMPLETE":
                break

            if action == "EXPAND":
                targets = assessment.get("target_aspects") or []
                if not targets:
                    targets = list(critical_gaps) if critical_gaps else list(uncovered)[:4]
                if not targets:
                    break
                slots = self.MAX_FINAL_QUESTIONS - len(questions)
                if slots <= 0:
                    break
                num_new = min(len(targets), slots, 4)
                new_qs = await self._expand(
                    query, query_type, domain, entity_instructions,
                    targets, num_new, clarified_topic, entity_a, entity_b
                )
                questions.extend(new_qs)
                questions = self._normalize_questions(
                    questions, query_type, domain, entity_a, entity_b, relationship, entity_c
                )
                self._log_questions(new_qs, f"EXPANDED (+{len(new_qs)})")

            elif action == "REFINE":
                indices = sorted(set(assessment.get("vague_indices", []) + assessment.get("redundant_indices", [])))
                if not indices:
                    indices = assessment.get("target_indices", [])
                if not indices:
                    break
                questions = await self._refine(query, query_type, domain, questions, indices, clarified_topic)
                questions = self._normalize_questions(
                    questions, query_type, domain, entity_a, entity_b, relationship, entity_c
                )
                self._log_questions(questions, "REFINED")

        covered, uncovered = _check_dimension_coverage(questions, query_type, domain)
        remaining_critical = _get_critical_set(query_type, domain) & uncovered

        if remaining_critical:
            gap_qs = await self._force_fill_gaps(
                query, query_type, domain, entity_instructions,
                remaining_critical, clarified_topic, entity_a=entity_a, entity_b=entity_b, entity_c=entity_c
            )
            questions.extend(gap_qs)
            questions = self._normalize_questions(questions, query_type, domain, entity_a, entity_b, relationship, entity_c)
            self._log_questions(gap_qs, "GAP FILLED")

        questions = self._normalize_questions(questions, query_type, domain, entity_a, entity_b, relationship, entity_c)

        questions = _strict_cosine_dedup(
            questions, query_type, domain, entity_a, entity_b, entity_c,
            similarity_threshold=0.65,   # FIX P2
        )

        if query_type == "comparison" and entity_a and entity_b:
            questions = _enforce_comparison_symmetry(
                questions, entity_a, entity_b, query_type, domain, entity_c,
                max_questions=self.MAX_FINAL_QUESTIONS,  # FIX P4
            )

        if len(questions) > self.MAX_FINAL_QUESTIONS:
            questions = _priority_trim(questions, self.MAX_FINAL_QUESTIONS, query_type, domain)

        if len(questions) < self.MIN_FINAL_QUESTIONS:
            questions = self._emergency_fallback(query, query_type, domain, self.MIN_FINAL_QUESTIONS, entity_a, entity_b, relationship, entity_c)

        covered_f, uncovered_f = _check_dimension_coverage(questions, query_type, domain)
        remaining_crit = _get_critical_set(query_type, domain) & uncovered_f

        logger.info(
            f"Planning COMPLETE | final_count={len(questions)} | dims_covered={len(covered_f)}/{len(dimensions)}"
        )
        if remaining_crit:
            logger.error(f"  CRITICAL dims still uncovered after all iterations: {', '.join(sorted(remaining_crit))}")
            rescue_qs = _build_template_questions(
                query_type, domain, remaining_crit, entity_a, entity_b,
                clarified_topic[:60], entity_c=entity_c
            )

            critical_set = _get_critical_set(query_type, domain)
            dims_list = _get_dimensions(query_type, domain)

            def _covers_critical_dim(q: str) -> bool:
                ql = _lower(q)
                tokens = _tokenize(q)
                for d in dims_list:
                    if d["key"] in critical_set:
                        sig = d.get("signals", set())
                        if sig & tokens or any(s in ql for s in sig):
                            return True
                return False

            for rq in rescue_qs:
                non_crit_indices = [i for i in range(len(questions) - 1, -1, -1)
                                    if not _covers_critical_dim(questions[i])]
                if non_crit_indices:
                    replace_idx = non_crit_indices[0]
                    logger.info(
                        f"  Critical gap fill: replacing Q{replace_idx+1}: '{questions[replace_idx][:60]}' "
                        f"with '{rq[:60]}'"
                    )
                    questions[replace_idx] = rq
                else:
                    questions.append(rq)

            questions = _semantic_deduplicate(
                questions,
                threshold=0.62,   # FIX P1 + P5: explicit threshold
                query_type=query_type,
                entity_a=entity_a,
                entity_b=entity_b,
                entity_c=entity_c,
            )
            if len(questions) > self.MAX_FINAL_QUESTIONS:
                questions = _priority_trim(questions, self.MAX_FINAL_QUESTIONS, query_type, domain)

            covered_f, uncovered_f = _check_dimension_coverage(questions, query_type, domain)
            remaining_crit = _get_critical_set(query_type, domain) & uncovered_f

        # Phase 13: Force key_differences for comparisons
        if (query_type == "comparison" and entity_a and entity_b
                and "key_differences" in uncovered_f):
            diff_q = (
                f"In what fundamental ways do {entity_a} and {entity_b}"
                + (f" and {entity_c}" if entity_c else "")
                + " differ in design, performance, and practical trade-offs?"
            )
            if len(questions) >= self.MAX_FINAL_QUESTIONS:
                questions[-1] = diff_q
            else:
                questions.append(diff_q)
            covered_f.add("key_differences")
            uncovered_f.discard("key_differences")
            remaining_crit.discard("key_differences")
            logger.info(f"  Phase 13 | Force-injected key_differences question")

        non_crit = uncovered_f - remaining_crit
        if non_crit:
            logger.info(f"  Non-critical uncovered: {', '.join(sorted(non_crit))}")

        self._log_questions(questions, "FINAL OUTPUT")

        planner_meta = {
            "query_type":     query_type,
            "domain":         domain,
            "entity_a":       entity_a,
            "entity_b":       entity_b,
            "entity_c":       entity_c,
            "relationship":   relationship if query_type == "comparison" else None,
            "covered_dims":   sorted(covered_f),
            "uncovered_dims": sorted(uncovered_f),
            "critical_gaps":  sorted(remaining_crit),
            "total_dims":     len(dimensions),
        }

        dim_keys = {d["key"] for d in dimensions}
        if covered_f | uncovered_f != dim_keys:
            missing = dim_keys - (covered_f | uncovered_f)
            extra   = (covered_f | uncovered_f) - dim_keys
            logger.error(
                f"FIX 4B | Dimension count mismatch! "
                f"missing_from_both={sorted(missing)} extra={sorted(extra)} "
                f"covered={len(covered_f)} uncovered={len(uncovered_f)} total={len(dimensions)}"
            )

        logger.info(
            f"  planner_meta | type={query_type} domain={domain} "
            f"covered={len(covered_f)}/{len(dimensions)} "
            f"critical_gaps={sorted(remaining_crit) if remaining_crit else 'none'}"
        )

        return {
            "sub_questions": questions,
            "planner_meta":  planner_meta,
        }

    def _disambiguate(self, query: str) -> str:
        return query

    def _build_entity_instructions(
        self,
        query_type: str,
        entity_a: Optional[str],
        entity_b: Optional[str],
        relationship: str,
        entity_c: Optional[str] = None,
    ) -> str:
        if query_type != "comparison" or not entity_a or not entity_b:
            return ""

        note = ""
        if relationship == "a_is_b":
            note = f"\nIMPORTANT: {entity_a} is a type/subclass of {entity_b}, not a peer."
        elif relationship == "b_is_a":
            note = f"\nIMPORTANT: {entity_b} is a type/subclass of {entity_a}, not a peer."

        if entity_c:
            return (
                f"Entity A: {entity_a}\n"
                f"Entity B: {entity_b}\n"
                f"Entity C: {entity_c}\n"
                f"BALANCE REQUIREMENT: cover all three entities equally — A, B, and C.\n"
                f"For differences, similarities, tradeoffs, limitations, and applications, "
                f"ensure all three entities are covered.\n"
                f"CRITICAL: Preserve acronym entities exactly as written."
                f"{note}"
            )

        return (
            f"Entity A: {entity_a}\n"
            f"Entity B: {entity_b}\n"
            f"BALANCE REQUIREMENT: alternate A/B coverage and keep both entities equally represented.\n"
            f"For definition, differences, similarities, tradeoffs, limitations, and applications, "
            f"generate separate questions for both entities when appropriate.\n"
            f"CRITICAL: Preserve acronym entities exactly as written unless the query explicitly expands them."
            f"{note}"
        )

    def _compute_initial_count(self, query_type: str) -> int:
        return {"comparison": 10, "deep_topic": 9, "how_to": 8, "factual": 6, "general": 9}.get(query_type, 8)

    def _log_questions(self, questions: List[str], label: str) -> None:
        logger.info(f"=== {label} ({len(questions)} questions) ===")
        for i, q in enumerate(questions, 1):
            logger.info(f"  Q{i}: {q}")

    def _comparison_semantic_priority(
        self, q: str, entity_a: Optional[str], entity_b: Optional[str],
        relationship: str, entity_c: Optional[str] = None
    ) -> Tuple[int, int]:
        qt = _lower(q)
        if any(k in qt for k in ["related", "relationship", "subset", "type of", "special case", "relate"]):
            pri = 0
        elif any(k in qt for k in ["what is", "definition", "overview", "define"]):
            pri = 1
        elif any(k in qt for k in ["difference", "compare", "contrast", "differ", "fundamental"]):
            pri = 2
        elif any(k in qt for k in ["similar", "shared", "common", "identify shared"]):
            pri = 3
        elif any(k in qt for k in ["tradeoff", "pros", "cons", "when should", "when to", "quantify"]):
            pri = 4
        elif any(k in qt for k in ["strength", "limitation", "weakness", "outline"]):
            pri = 5
        elif any(k in qt for k in ["application", "use case", "deployment", "deployment scenario"]):
            pri = 6
        elif any(k in qt for k in ["benchmark", "performance", "evaluation", "latency", "speed", "cost", "compare inference"]):
            pri = 7
        elif any(k in qt for k in ["recent", "future", "summarize", "open research"]):
            pri = 8
        else:
            pri = 9

        tag = _entity_tag(q, entity_a, entity_b, entity_c)
        entity_bias = 0 if tag == "a" else 1 if tag == "b" else 2 if tag == "c" else 3
        return pri, entity_bias

    def _prioritize_questions(
        self,
        questions: List[str],
        query_type: str,
        domain: str,
        entity_a: Optional[str] = None,
        entity_b: Optional[str] = None,
        relationship: str = "peer",
        entity_c: Optional[str] = None,
    ) -> List[str]:
        if query_type != "comparison":
            def score(q: str) -> int:
                qt = _lower(q)
                if any(k in qt for k in ["what is", "definition", "define"]):
                    return 0
                if any(k in qt for k in ["how", "mechanism"]):
                    return 1
                if any(k in qt for k in ["difference", "compare", "contrast"]):
                    return 2
                if "tradeoff" in qt:
                    return 3
                if any(k in qt for k in ["application", "use case"]):
                    return 4
                if any(k in qt for k in ["limitation", "weakness"]):
                    return 5
                if any(k in qt for k in ["recent", "future"]):
                    return 6
                return 7
            return sorted(questions, key=score)

        return sorted(questions, key=lambda q: self._comparison_semantic_priority(q, entity_a, entity_b, relationship, entity_c))

    # FIX P5: _normalize_questions now always passes threshold=0.62 explicitly
    def _normalize_questions(
        self,
        questions: List[str],
        query_type: str,
        domain: str,
        entity_a: Optional[str],
        entity_b: Optional[str],
        relationship: str,
        entity_c: Optional[str] = None,
    ) -> List[str]:
        qs = [q.strip() for q in questions if _is_valid_question(q)]
        # FIX P5: explicit threshold — was using default 0.72 from the function signature
        qs = _semantic_deduplicate(
            qs, threshold=0.62,
            query_type=query_type, entity_a=entity_a, entity_b=entity_b, entity_c=entity_c
        )

        if query_type == "comparison" and entity_a and entity_b:
            qs = _comparison_balancing_cleanup(qs, entity_a, entity_b, relationship, entity_c)

        qs = self._prioritize_questions(qs, query_type, domain, entity_a, entity_b, relationship, entity_c)
        # FIX P5: second pass also uses explicit threshold
        return _semantic_deduplicate(
            qs, threshold=0.62,
            query_type=query_type, entity_a=entity_a, entity_b=entity_b, entity_c=entity_c
        )

    async def _generate_initial(
        self,
        query: str,
        query_type: str,
        domain: str,
        entity_instructions: str,
        count: int,
        required_dims_str: str,
        clarified_topic: str,
        entity_a: Optional[str],
        entity_b: Optional[str],
    ) -> List[str]:
        try:
            resp = await _llm_invoke(
                self.generation_chain,
                {
                    "query": query,
                    "query_type": query_type,
                    "domain": domain,
                    "entity_instructions": entity_instructions,
                    "initial_count": count,
                    "required_dimensions": required_dims_str,
                    "clarified_topic": clarified_topic,
                },
                label="generate_initial",
            )
            qs = _parse_questions(resp.content)
            if not qs:
                raise ValueError("No valid questions parsed")
            self._consecutive_errors = 0
            return qs
        except Exception as e:
            self._consecutive_errors += 1
            if "429" in str(e) or "quota" in str(e).lower():
                self._quota_exhausted = True
            logger.error(f"Initial generation failed: {e}")
            if query_type == "comparison" and entity_a and entity_b:
                return _comparison_seed_questions(entity_a, entity_b, "peer", domain)
            return self._emergency_fallback(query, query_type, domain, count, entity_a, entity_b, "peer")

    async def _assess(
        self,
        query: str,
        query_type: str,
        domain: str,
        questions: List[str],
        covered: Set[str],
        uncovered: Set[str],
        critical_gaps: Set[str],
        cov_ratio: float,
        clarified_topic: str,
    ) -> Dict:
        if self._quota_exhausted:
            return {
                "action": "EXPAND",
                "target_aspects": list(critical_gaps) if critical_gaps else list(uncovered)[:4],
                "vague_indices": [],
                "redundant_indices": [],
                "target_indices": [],
            }

        questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        try:
            resp = await _llm_invoke(
                self.assess_chain,
                {
                    "query": query,
                    "query_type": query_type,
                    "domain": domain,
                    "questions": questions_str,
                    "covered_dimensions": ", ".join(sorted(covered)) or "none",
                    "uncovered_dimensions": ", ".join(sorted(uncovered)) or "none",
                    "critical_gaps": ", ".join(sorted(critical_gaps)) or "none",
                    "coverage_ratio": f"{cov_ratio:.2f}",
                },
                label="assess",
            )
            result = _extract_json(resp.content)
            if not result:
                raise ValueError("Empty assessment JSON")
            self._consecutive_errors = 0
            logger.info(
                f"Assessment | coverage={result.get('coverage_score')}/10 quality={result.get('quality_score')}/10 action={result.get('action')}"
            )
            return result
        except Exception as e:
            self._consecutive_errors += 1
            if "429" in str(e) or "quota" in str(e).lower():
                self._quota_exhausted = True
            logger.warning(f"Assessment failed: {e} — defaulting EXPAND")
            return {
                "action": "EXPAND",
                "target_aspects": list(critical_gaps) if critical_gaps else list(uncovered)[:4],
                "vague_indices": [],
                "redundant_indices": [],
                "target_indices": [],
            }

    async def _expand(
        self,
        query: str,
        query_type: str,
        domain: str,
        entity_instructions: str,
        target_aspects: List[str],
        num_new: int,
        clarified_topic: str,
        entity_a: Optional[str],
        entity_b: Optional[str],
    ) -> List[str]:
        if self._quota_exhausted:
            return _build_template_questions(query_type, domain, set(target_aspects), entity_a, entity_b, clarified_topic[:60])

        aspects_readable = _dimension_keys_to_descriptions(set(target_aspects), query_type, domain)
        try:
            resp = await _llm_invoke(
                self.expand_chain,
                {
                    "query": query,
                    "query_type": query_type,
                    "domain": domain,
                    "entity_instructions": entity_instructions,
                    "target_aspects": aspects_readable,
                    "num_new": num_new,
                    "clarified_topic": clarified_topic,
                },
                label="expand",
            )
            self._consecutive_errors = 0
            return _parse_questions(resp.content)[:num_new]
        except Exception as e:
            self._consecutive_errors += 1
            if "429" in str(e) or "quota" in str(e).lower():
                self._quota_exhausted = True
            logger.error(f"Expansion failed: {e}")
            return _build_template_questions(query_type, domain, set(target_aspects), entity_a, entity_b, clarified_topic[:60])

    async def _refine(
        self,
        query: str,
        query_type: str,
        domain: str,
        questions: List[str],
        target_indices: List[int],
        clarified_topic: str,
    ) -> List[str]:
        if self._quota_exhausted:
            return questions

        valid = [i for i in target_indices if 0 <= i < len(questions)]
        if not valid:
            return questions

        to_refine = "\n".join(f"{i+1}. {questions[i]}" for i in valid)
        try:
            resp = await _llm_invoke(
                self.refine_chain,
                {
                    "query": query,
                    "query_type": query_type,
                    "domain": domain,
                    "questions_to_refine": to_refine,
                    "clarified_topic": clarified_topic,
                },
                label="refine",
            )
            refined_map: Dict[int, str] = {}
            for line in resp.content.strip().splitlines():
                m = re.match(r"^(\d+)[\.\)]\s+(.+)$", line.strip())
                if m:
                    idx = int(m.group(1)) - 1
                    text = m.group(2).strip()
                    if 0 <= idx < len(questions) and _is_valid_question(text):
                        refined_map[idx] = text

            if not refined_map:
                refined_raw = _parse_questions(resp.content)
                for pos, idx in enumerate(valid):
                    if pos < len(refined_raw):
                        refined_map[idx] = refined_raw[pos]

            result = list(questions)
            for idx, text in refined_map.items():
                logger.info(f"  Refined Q{idx+1}: '{questions[idx]}' → '{text}'")
                result[idx] = text
            self._consecutive_errors = 0
            return result
        except Exception as e:
            self._consecutive_errors += 1
            if "429" in str(e) or "quota" in str(e).lower():
                self._quota_exhausted = True
            logger.warning(f"Refinement failed: {e} — keeping originals")
            return questions

    async def _force_fill_gaps(
        self,
        query: str,
        query_type: str,
        domain: str,
        entity_instructions: str,
        gaps: Set[str],
        clarified_topic: str,
        entity_a: Optional[str] = None,
        entity_b: Optional[str] = None,
        entity_c: Optional[str] = None,
        relationship: str = "peer",
    ) -> List[str]:
        if not gaps:
            return []

        if self._quota_exhausted or self._consecutive_errors >= self._API_ERROR_LIMIT:
            logger.warning("Gap fill: quota exhausted — using template fallback")
            return _build_template_questions(query_type, domain, gaps, entity_a, entity_b, clarified_topic[:60], entity_c=entity_c)

        gap_dims = _get_dimensions(query_type, domain)
        gap_map = {d["key"]: d["desc"] for d in gap_dims}
        gap_list = "\n".join(f"{i+1}. [{k}]: {gap_map.get(k, k)}" for i, k in enumerate(sorted(gaps)))

        try:
            resp = await _llm_invoke(
                self.gap_fill_chain,
                {
                    "query": query,
                    "query_type": query_type,
                    "domain": domain,
                    "entity_instructions": entity_instructions,
                    "gap_list": gap_list,
                    "num_gaps": len(gaps),
                    "clarified_topic": clarified_topic,
                },
                label="gap_fill",
            )
            self._consecutive_errors = 0
            return _parse_questions(resp.content)[: len(gaps)]
        except Exception as e:
            self._consecutive_errors += 1
            if "429" in str(e) or "quota" in str(e).lower():
                self._quota_exhausted = True
            logger.error(f"Gap fill LLM failed: {e} — template fallback")
            return _build_template_questions(query_type, domain, gaps, entity_a, entity_b, clarified_topic[:60], entity_c=entity_c)

    def _emergency_fallback(
        self,
        query: str,
        query_type: str,
        domain: str,
        n: int,
        entity_a: Optional[str] = None,
        entity_b: Optional[str] = None,
        relationship: str = "peer",
        entity_c: Optional[str] = None,
    ) -> List[str]:
        topic = _normalize_text(query)[:60].strip()

        if query_type == "comparison" and entity_a and entity_b:
            if entity_c:
                base = [
                    f"Define {entity_a}: core concept, purpose, and how it operates.",
                    f"Define {entity_b}: core concept, purpose, and how it operates.",
                    f"Define {entity_c}: core concept, purpose, and how it operates.",
                    f"How do {entity_a}, {entity_b}, and {entity_c} relate to each other?",
                    f"Identify shared characteristics of {entity_a}, {entity_b}, and {entity_c}.",
                    f"In what fundamental ways do {entity_a}, {entity_b}, and {entity_c} differ?",
                    f"Outline the strengths of {entity_a}, {entity_b}, and {entity_c}.",
                    f"Outline the limitations of {entity_a}, {entity_b}, and {entity_c}.",
                    f"Quantify the tradeoffs between {entity_a}, {entity_b}, and {entity_c}.",
                    f"When should each of {entity_a}, {entity_b}, {entity_c} be chosen?",
                    f"Compare benchmarks: how do {entity_a}, {entity_b}, and {entity_c} score?",
                    f"Summarize future directions for {entity_a}, {entity_b}, and {entity_c}.",
                ]
            else:
                if relationship == "a_is_b":
                    rel_q = f"In what sense is {entity_a} a specialized form of {entity_b}?"
                elif relationship == "b_is_a":
                    rel_q = f"In what sense is {entity_b} a specialized form of {entity_a}?"
                else:
                    rel_q = f"How do {entity_a} and {entity_b} relate — complementary, competing, or orthogonal?"
                base = [
                    f"Define {entity_a}: core purpose, design, and internal mechanism.",
                    f"Define {entity_b}: core purpose, design, and internal mechanism.",
                    rel_q,
                    f"Identify shared characteristics that {entity_a} and {entity_b} have in common.",
                    f"In what fundamental ways do {entity_a} and {entity_b} differ?",
                    f"Outline known limitations and failure modes of {entity_a}.",
                    f"Outline known limitations and failure modes of {entity_b}.",
                    f"Quantify the tradeoffs between {entity_a} and {entity_b}.",
                    f"Explain the primary real-world deployment scenarios for {entity_a} and {entity_b}.",
                    f"When should {entity_a} be chosen over {entity_b}, and vice versa?",
                    f"Compare {entity_a} and {entity_b} on standard performance benchmarks.",
                    f"Summarize open research problems and future directions for {entity_a} and {entity_b}.",
                ]
            return _semantic_deduplicate(
                base, threshold=0.62,
                query_type="comparison",
                entity_a=entity_a, entity_b=entity_b, entity_c=entity_c
            )[:n]

        if query_type == "how_to":
            return [
                f"What is {topic} and what prerequisites are needed?",
                f"Explain how {topic} works at its core mechanism level.",
                f"List the tools, libraries, or frameworks required for {topic}.",
                f"Describe the step-by-step implementation guide for {topic}.",
                f"Explain the real-world use cases and examples of {topic}.",
                f"Outline common pitfalls and limitations when implementing {topic}.",
                f"How do you optimise performance and efficiency for {topic}?",
                f"Provide sample code implementations and case studies for {topic}.",
            ][:n]

        if query_type == "factual":
            return [
                f"Define {topic} formally and explain its core purpose.",
                f"Trace the history and origin of {topic}.",
                f"Explain how {topic} works at its core mechanism.",
                f"Describe the main types and variants of {topic}.",
                f"List the real-world applications of {topic}.",
                f"Outline the main limitations or challenges of {topic}.",
            ][:n]

        return [
            f"Define {topic} and explain its core purpose formally.",
            f"Explain how {topic} works at its core mechanism level.",
            f"Trace the history and origin of {topic}.",
            f"Describe the main types and variants of {topic}.",
            f"List the practical applications of {topic}.",
            f"Summarise benchmark and performance results for {topic}.",
            f"Outline the known limitations and challenges of {topic}.",
            f"Describe the most recent advances in {topic}.",
            f"Outline the future research directions in {topic}.",
        ][:n]


__all__ = [
    "PlannerAgent",
    "extract_comparison_entities",
    "detect_query_type",
    "detect_domain_family",
    "normalize_acronyms",
]