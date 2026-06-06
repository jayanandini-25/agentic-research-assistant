from __future__ import annotations

import asyncio
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config.settings import get_settings
from core.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_SEARCH_URL  = "https://commons.wikimedia.org/w/api.php"

GEMINI_TEXT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp"}

STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "vs", "versus", "for", "to",
    "is", "are", "was", "were", "how", "what", "why", "do", "does", "between",
    "on", "at", "with", "from", "that", "this", "these", "those", "compare",
    "comparison", "differences", "difference", "contrast", "contrasting",
    "best", "good", "top", "latest", "new", "overview", "explain", "image",
    "picture", "show", "give", "me", "please", "find", "looking", "need",
    "key", "main", "important", "recent", "advances", "should", "when",
    "surpass", "terms", "similar", "similarities", "tradeoffs", "preferred",
    "real", "world", "use", "cases", "strengths", "limitations", "related",
    "primary", "faced", "low", "quality", "degrade",
}

MIN_MEANINGFUL_TERMS = 1
MIN_TERM_HITS        = 1
MIN_SCORE            = 0.38
MIN_GEMINI           = 0.50  # FIX: raised from 0.45 — tighter Gemini gate

# FIX: Tightened per-source minimum score thresholds across the board.
# meta_image raised to 0.80 (was 0.70) — stock photos/cartoons were still slipping through.
# page_img raised to 0.50 (was 0.42) — non-research page images too noisy.
SOURCE_MIN_SCORE: Dict[str, float] = {
    "arxiv":      0.38,
    "wikimedia":  0.38,
    "wikipedia":  0.40,
    "page_img":   0.50,   # raised from 0.42
    "body_img":   0.48,   # raised from 0.45
    "meta_image": 0.80,   # raised from 0.70 — cartoons/stock photos need much higher bar
    "openverse":  0.40,
}

_DISCARD_URL_PATTERNS = {
    "/wp-content/uploads/", "/assets/img/blog/", "/assets/images/header",
    "/static/img/banner", "/media/hero", "/newsletter",
    "/promo/", "/ads/", "/marketing/", "/campaign/",
    "stock-photo", "generic-image", "placeholder",
    "abstract-background", "geometric-pattern", "/decorative/",
    "/hero-bg", "/bg-image", "pattern-overlay",
}

# FIX: Massively expanded _DISCARD_ALT_PATTERNS.
# Root cause of comic/stock-photo slipping through: their alt text was the
# article TITLE (e.g. "Fine-tuning small language models has been gaining
# traction...") which contained the query term "fine-tuning" so it scored
# high on term_overlap. We now explicitly block:
# 1. Alt text that reads like a blog article headline
# 2. Alt text referencing cartoons/comics/illustrations
# 3. Common stock-photo descriptions
_DISCARD_ALT_PATTERNS = {
    # People / office
    "author", "headshot", "team member", "contributor",
    "sponsored", "advertisement", "ad banner", "promotion",
    "subscribe", "newsletter", "signup", "sign up",
    "product shot", "company logo", "brand",
    "abstract background", "geometric pattern", "decorative",
    "abstract shapes", "colorful background", "gradient background",
    # Marketing blog OG titles
    "unlock ai",
    "full potential",
    "power of fine",
    "compare fine-tuning vs",
    "understand the difference",
    "when and how to",
    "learn how",
    "discover how",
    "use cases of",
    # FIX: Comic/cartoon/illustration detection
    "cartoon",
    "comic",
    "illustration",
    "clip art",
    "clipart",
    "meme",
    "funny",
    "humor",
    "i'm b-tuning",          # the actual comic caption seen in screenshot
    "did you try prompt",    # the actual comic caption seen in screenshot
    "playing violin",
    "musical note",
    # FIX: Stock-photo / lifestyle phrases that appear in OG descriptions
    "gaining traction",
    "has been gaining",
    "enterprise ai teams",
    "many enterprise",
    "when an llm doesn",
    "large language models are trained",
    "adapt pretrained models",        # "Discover AI fine-tuning and model tuning to adapt pretrained models"
    "model tuning to adapt",
    "pretrained models in...",
    # FIX: Generic "what is X" blog thumbnails
    "stands out as a groundb",        # "Retrieval-Augmented Generation (RAG) stands out as a groundbreaking..."
    "groundbreaking",
}

# FIX: Regex to catch office/people/lifestyle photos — expanded significantly.
_DISCARD_ALT_PEOPLE = re.compile(
    r"\b(people|person|team|office|meeting|interview|hands|desk|"
    r"laptop|computer\s+screen|professional|employee|worker|startup|"
    r"colleagues|conference|handshake|boardroom|"
    # FIX additions:
    r"woman|man|girl|boy|student|instructor|teacher|coach|"
    r"sitting|standing|looking\s+at|pointing\s+at|"
    r"three\s+people|two\s+people|group\s+of)\b",
    re.IGNORECASE,
)

# FIX: New regex to catch cartoon/comic/illustration images by URL pattern.
_DISCARD_URL_CARTOON = re.compile(
    r"(cartoon|comic|clipart|clip-art|illustration|meme|"
    r"funny|humor|drawing|sketch|toon|animation)",
    re.IGNORECASE,
)

_IMAGE_AI_ML_TERMS = {
    "neural", "network", "transformer", "attention", "model",
    "architecture", "diffusion", "gan", "generative", "embedding",
    "training", "inference", "pipeline", "rag", "retrieval",
    "language", "bert", "gpt", "llm", "benchmark", "encoder",
    "decoder", "layer", "gradient", "loss", "optimization",
    "classification", "detection", "segmentation", "diagram",
    "figure", "chart", "comparison", "performance", "accuracy",
}

_IMAGE_AI_ML_QUERY_SIGNALS = {
    "machine learning", "deep learning", "neural network", "transformer",
    "llm", "language model", "diffusion model", "gan", "generative",
    "rag", "retrieval-augmented", "artificial intelligence",
    "bert", "gpt", "computer vision",
}

# FIX: Alt text patterns that indicate a TECHNICAL DIAGRAM (good images).
# Used to allow low-score arxiv/wikipedia images that would otherwise be cut.
_DIAGRAM_ALT_SIGNALS = re.compile(
    r"\b(diagram|architecture|pipeline|workflow|figure|overview|"
    r"comparison|benchmark|flow|schema|illustration\s+of|structure|"
    r"rag\s+(diagram|process|pipeline|overview)|"
    r"fine.tun|retrieval.augmented)\b",
    re.IGNORECASE,
)


def _image_is_ai_ml_query(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in _IMAGE_AI_ML_QUERY_SIGNALS)


def _post_score_discard(img_url: str, alt: str) -> bool:
    """FIX: Extended to also check for cartoon URLs and expanded alt patterns."""
    url_lower = img_url.lower()
    alt_lower = alt.lower()
    if any(p in url_lower for p in _DISCARD_URL_PATTERNS):
        return True
    if _DISCARD_URL_CARTOON.search(url_lower):
        return True
    if any(p in alt_lower for p in _DISCARD_ALT_PATTERNS):
        return True
    return False


def _is_clearly_not_diagram(img_url: str, alt: str, source: str) -> bool:
    """
    FIX: New hard filter — for AI/ML queries, meta_image and page_img sources
    MUST have diagram-like signals in their alt text or URL, otherwise reject.
    
    This catches the office photo ("Discover AI fine-tuning...") and the comic
    ("Fine-tuning small language models has been gaining traction...") which
    both had plausible alt text but zero diagram signals.
    """
    if source not in ("meta_image", "page_img", "body_img"):
        return False  # arxiv/wikipedia/wikimedia exempt — they're pre-vetted sources
    
    combined = f"{img_url} {alt}".lower()
    has_diagram_signal = _DIAGRAM_ALT_SIGNALS.search(combined)
    return not has_diagram_signal


SOURCE_TIMEOUT   = 15
COLLECT_TIMEOUT  = 30
GEMINI_TIMEOUT   = 10

BLOG_HOSTS = {
    "medium.com", "miro.medium.com", "cdn.hashnode.com",
    "dev.to", "ghost.io", "substack.com",
    "towardsdatascience.com", "analyticsvidhya.com",
    "neptune.ai", "tryagi.com", "vectara.com",
    "pinecone.io", "weaviate.io", "qdrant.io",
    "langchain.com", "llamaindex.ai",
    "datastax.com", "mongodb.com",
    "betterprogramming.pub", "levelup.gitconnected.com",
    "hackernoon.com", "dzone.com",
    "machinelearningmastery.com",
    "cloud.google.com", "azure.microsoft.com", "aws.amazon.com",
    "datacamp.com", "kdnuggets.com",
    "builtin.com", "simplilearn.com", "geeksforgeeks.org",
    "oracle.com", "ibm.com", "coursera.org", "linearloop.io",
    "pecollective.com", "digitalapplied.com", "anolytics.ai",
    "stack-ai.com", "coreweave.com",
    # FIX: Additional blog/marketing domains that generate cartoon/stock images
    "solutelabs.com", "therightsw.com", "protecto.ai",
    "mindbreeze.com", "encord.com", "blogs.nvidia.com",
}

BLOCKED_PATTERNS = {
    "logo", "icon", "favicon", "spinner", "avatar", "badge", "button",
    "pixel", "tracking", "captcha", "placeholder", "blank", "spacer",
    "1x1", "transparent", "base64", "share-image",
    "author-photo", "author-avatar", "profile-pic", "newsletter", "subscribe",
    "promo", "banner-ad", "watermark", "hero-image", "cover-image",
    "shutterstock", "gettyimages", "istockphoto", "depositphotos",
    "unsplash", "pexels", "pixabay",
    # FIX: common cartoon/comic CDN patterns
    "cartoon", "comic", "clipart",
}

TRUSTED_HOSTS = {
    "arxiv.org", "wikipedia.org", "upload.wikimedia.org",
    "commons.wikimedia.org", "nature.com", "springer.com",
    "ieee.org", "acm.org", "openai.com", "huggingface.co",
    "deepmind.com", "anthropic.com", "research.google",
    "ai.googleblog.com", "ai.meta.com", "proceedings.mlr.press",
    "papers.nips.cc", "distill.pub", "cs.stanford.edu",
    "cs.berkeley.edu", "cs.cmu.edu", "openreview.net",
    "semanticscholar.org", "aclanthology.org",
}

WIKIPEDIA_TOPIC_MAP = {
    "transformer": ["Transformer_(deep_learning_architecture)", "Attention_Is_All_You_Need"],
    "llm": ["Large_language_model", "Generative_pre-trained_transformer"],
    "large language": ["Large_language_model", "GPT-4"],
    "neural network": ["Artificial_neural_network", "Deep_learning"],
    "deep learning": ["Deep_learning", "Convolutional_neural_network"],
    "machine learning": ["Machine_learning", "Supervised_learning"],
    "bert": ["BERT_(language_model)"],
    "gpt": ["Generative_pre-trained_transformer", "GPT-4"],
    "attention": ["Transformer_(deep_learning_architecture)"],
    "nlp": ["Natural_language_processing"],
    "rag": ["Retrieval-augmented_generation", "Information_retrieval"],
    "retrieval-augmented": ["Retrieval-augmented_generation"],
    "retrieval augmented": ["Retrieval-augmented_generation"],
    "fine-tuning": ["Fine-tuning_(deep_learning)"],
    "fine tuning": ["Fine-tuning_(deep_learning)"],
    "embedding": ["Word_embedding", "Sentence_embedding"],
    "reinforcement": ["Reinforcement_learning", "Q-learning"],
    "computer vision": ["Computer_vision", "Convolutional_neural_network"],
    "diffusion": ["Diffusion_model", "Stable_Diffusion"],
    "gan": ["Generative_adversarial_network"],
    "generative adversarial": ["Generative_adversarial_network"],
    "graph database": ["Graph_database", "Neo4j"],
    "vector database": ["Vector_database", "Approximate_nearest_neighbor_search"],
}

ACRONYM_MAP = {
    "rag": "Retrieval-Augmented Generation",
    "llm": "Large Language Model",
    "llms": "Large Language Models",
    "gpt": "Generative Pre-trained Transformer",
    "bert": "Bidirectional Encoder Representations from Transformers",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "ai": "Artificial Intelligence",
    "gan": "Generative Adversarial Network",
    "gans": "Generative Adversarial Networks",
}

SOURCE_BOOST = {
    "arxiv":      0.25,
    "wikimedia":  0.20,
    "wikipedia":  0.18,
    "openverse":  0.05,
    "meta_image": -0.25,  # FIX: penalty increased from -0.20 to -0.25
    "page_img":    0.00,
    "body_img":   -0.15,
}

# FIX: meta_image hard cap reduced to 0 for AI/ML queries (set dynamically).
# For non-AI/ML queries, keep at 1. This is handled in _apply_caps.
# Global default caps:
SOURCE_CAP = {
    "arxiv":      6,
    "wikimedia":  5,
    "wikipedia":  4,
    "openverse":  3,
    "meta_image": 1,   # absolute max; overridden to 0 for AI/ML queries
    "page_img":   3,   # reduced from 4
    "body_img":   2,
}

_MIN_ALT_LEN: Dict[str, int] = {
    "meta_image": 25,   # FIX: raised from 20
    "page_img":   15,   # FIX: raised from 12
    "body_img":   25,
    "arxiv":       8,
    "wikipedia":   5,
    "wikimedia":   5,
    "openverse":   5,
}

_JUNK_ALT_PATTERNS = re.compile(
    r"^(figure\s*\d+\.?|fig\.?\s*\d+|image\s*\d+|photo\s*\d+|"
    r"click\s*(here|to\s+use)|untitled|placeholder|hero\s+image|"
    r"article\s+hero|header\s+image|thumbnail|logo|icon|avatar|"
    r"refer\s+to\s+caption|see\s+caption|caption\s+below|"
    r"data\s+data|stock\s+photo|getty|shutterstock|"
    r"blog\s+(header|cover|image)|cover\s+image|"
    r"illustration|decorative|abstract\s+background)$",
    re.IGNORECASE,
)

_GENERIC_MARKETING_ALT = re.compile(
    r"^(large\s+language\s+models?\s+(llms?\s+)?are\s+trained|"
    r"when\s+an\s+llm\s+doesn.t\s+meet|"
    r"many\s+enterprise\s+ai\s+teams|"
    r"discover\s+how|learn\s+how|"
    r"unlock\s+ai|"
    # FIX: additional marketing alt patterns
    r"discover\s+ai\s+fine.tun|"       # "Discover AI fine-tuning..."
    r"fine.tuning\s+small\s+language|" # "Fine-tuning small language models..."
    r"\w+\s+blog$|by\s+\w+\s+\w+$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Short query extraction
# ---------------------------------------------------------------------------

def _extract_short_query(question: str) -> str:
    q = re.sub(r"\?$", "", question.strip()).lower()
    q = re.sub(
        r"^(what (are|is|were)|how (do|does|did|are)|when (should|do|does)|"
        r"why (are|do|does)|which|where|who)\s+",
        "", q, flags=re.I,
    )
    for phrase in [
        "the key ", "the main ", "the most important ", "the recent ",
        "real-world ", "in terms of ", "compared to ", "compared with ",
        "surpass gans in ", "that surpass ", "be preferred over ",
        "should be preferred", "key differences between ", "differences between ",
        "similarities between ", "tradeoffs between ", "strengths of ",
        "limitations of ", "use cases for ", "advances in ",
        "recent advances in ", "performance benchmarks", "primary tradeoffs between ",
        "ability to leverage ", "reliance on ", "handling out-of-domain ",
        "performance degrade when faced with ", "low-quality retrieval results",
        "challenges in implementing ", "advantages of ", "over fine-tuning",
        "key advantages of ", "ability leverage external knowledge compare ",
    ]:
        q = q.replace(phrase, " ")
    q = re.sub(r"\s+", " ", q).strip()
    words = [w for w in q.split() if w not in STOP_WORDS and len(w) > 2]
    short = " ".join(words[:5])
    if len(short) < 5:
        orig_words = [w for w in re.findall(r"\b\w+\b", question.lower())
                      if w not in STOP_WORDS and len(w) > 2]
        short = " ".join(orig_words[:4])
    return short.strip() or question[:40]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = (text or "").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def _lower(text: str) -> str:
    return _normalize_text(text).lower()


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[\w-]+\b", _lower(text))
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def _meaningful_terms(query: str) -> List[str]:
    terms = []
    for t in _tokenize(query):
        if t not in terms:
            terms.append(t)
    return terms


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _is_trusted(url: str) -> bool:
    host = _domain(url)
    return any(h in host for h in TRUSTED_HOSTS)


def _is_blog_host(url: str) -> bool:
    host = _domain(url)
    return any(b in host for b in BLOG_HOSTS)


def _is_blocked(url: str, text: str = "") -> bool:
    try:
        combined = f"{url} {text}".lower()
        return any(p in combined for p in BLOCKED_PATTERNS)
    except Exception:
        return True


def _is_valid_image_url(url: str) -> bool:
    try:
        normalized = _normalize_url(url)
        path = urlparse(normalized).path.lower()
        if not path:
            return False
        if any(bad in path for bad in (
            "placeholder", "blank", "default", "unavailable", "missing",
            "pixel", "spacer", "tracking", "avatar", "logo", "icon",
        )):
            return False
        if any(path.endswith(ext) for ext in VALID_EXTENSIONS):
            return True
        last = path.split("/")[-1] if "/" in path else path
        if not last or len(last) <= 3:
            return False
        if any(x in last for x in ("figure", "img", "image", "photo", "pic", "thumb")):
            return True
        return False
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    if not url:
        return url
    try:
        url = urllib.parse.unquote(url)
        url = re.sub(r"[?&](w|h|width|height|size|quality|fit|auto|format)=\d+", "", url)
        url = re.sub(r"[?&](download|raw)=1", "", url)
        return url.rstrip("?&")
    except Exception:
        return url


def _url_path_key(url: str) -> str:
    try:
        path = urlparse(_normalize_url(url)).path
        basename = path.split("/")[-1]
        basename = re.sub(r"^\d+px-", "", basename)
        return basename.lower()
    except Exception:
        return url


def _fallback_image_label(url: str, source: str = "") -> str:
    try:
        basename = urlparse(_normalize_url(url)).path.split("/")[-1]
        basename = re.sub(r"\.(jpg|jpeg|png|webp|gif|svg|bmp)$", "", basename, flags=re.I)
        basename = re.sub(r"^\d+px-", "", basename)
        basename = re.sub(r"[-_]+", " ", basename).strip()
        basename = re.sub(r"\s+", " ", basename)
        if basename and len(basename) > 3:
            return basename[:120]
    except Exception:
        pass
    label = (source or "image").replace("_", " ").strip().title()
    return f"{label} image"


def _select_best_srcset_url(srcset: str, base_url: str) -> Optional[str]:
    best_url: Optional[str] = None
    best_score = -1.0
    if not srcset:
        return None
    for chunk in srcset.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        raw = parts[0].strip()
        if not raw or raw.startswith("data:"):
            continue
        score = 1.0
        if len(parts) > 1:
            desc = parts[1].strip().lower()
            m = re.match(r"(\d+(?:\.\d+)?)(w|x)$", desc)
            if m:
                score = float(m.group(1))
                if m.group(2) == "x":
                    score *= 1000.0
        url = _extract_img_url(raw, base_url)
        if url and _is_valid_image_url(url) and score > best_score:
            best_score = score
            best_url = url
    return best_url


def _extract_img_url_from_tag(img_tag, base_url: str) -> Optional[str]:
    if img_tag is None:
        return None
    for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-fallback-src"):
        val = img_tag.get(attr, "")
        if val and not str(val).startswith("data:"):
            url = _extract_img_url(str(val), base_url)
            if url and _is_valid_image_url(url):
                return url
    for attr in ("srcset", "data-srcset", "data-original-srcset"):
        val = img_tag.get(attr, "")
        url = _select_best_srcset_url(str(val), base_url)
        if url:
            return url
    return None


def _intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in [
        "diagram", "architecture", "pipeline", "workflow", "system",
        "structure", "flow", "figure", "chart", "network", "model",
        "block diagram", "process", "mechanism", "framework",
    ]):
        return "diagram"
    if any(k in q for k in [
        "who is", "biography", "portrait", "profile", "city", "country",
        "travel", "landmark", "animal", "plant", "building",
    ]):
        return "photo"
    return "general"


def _alt_is_junk(alt: str, source: str) -> bool:
    alt = alt.strip()
    if not alt:
        return True
    if len(alt) < _MIN_ALT_LEN.get(source, 8):
        return True
    if _JUNK_ALT_PATTERNS.match(alt):
        return True
    if source in ("meta_image", "page_img") and _GENERIC_MARKETING_ALT.match(alt):
        return True
    return False


def _term_overlap(query: str, *text_fields: str) -> Tuple[int, float]:
    terms = _meaningful_terms(query)
    if not terms:
        return 0, 0.0
    combined = " ".join(f.lower() for f in text_fields)
    hits = sum(1 for t in terms if t in combined)
    return hits, hits / max(len(terms), 1)


def _expand_acronyms(text: str) -> str:
    out = _normalize_text(text)
    for short, long in ACRONYM_MAP.items():
        out = re.sub(rf"\b{re.escape(short)}\b", f"{long} ({short.upper()})", out, flags=re.I)
    return out


def _source_query_variants(short_query: str) -> Dict[str, List[str]]:
    q = _normalize_text(short_query)
    is_comparison = bool(re.search(r"\b(vs\.?|versus|compared|difference|differences)\b", q, re.I))
    terms = _meaningful_terms(short_query)

    if len(terms) < MIN_MEANINGFUL_TERMS and not is_comparison:
        return {k: [] for k in ["wikipedia", "wikimedia", "arxiv"]}

    expanded = _expand_acronyms(short_query)
    if is_comparison:
        parts = re.split(r"\s+(?:vs\.?|versus|and)\s+", q, flags=re.I)
        if len(parts) >= 2:
            a, b = parts[0].strip(), parts[1].strip()
            return {
                "wikipedia": [a, b, short_query],
                "wikimedia": [short_query, a, b],
                "arxiv":     [short_query, a, b],
            }
    return {
        "wikipedia": [short_query, expanded],
        "wikimedia": [short_query, expanded],
        "arxiv":     [short_query, expanded],
    }


# ---------------------------------------------------------------------------
# CandidateImage
# ---------------------------------------------------------------------------

@dataclass
class CandidateImage:
    url:        str
    alt:        str
    source:     str
    page_url:   str
    score:      float
    confidence: str = "medium"
    width:      int = 0
    height:     int = 0
    caption:    str = ""
    scored_by:  str = "lexical"
    domain:     str = field(default_factory=str)
    term_hits:  int = 0
    is_blog:    bool = False

    def to_dict(self) -> Dict:
        display_alt = self.alt or self.caption or _fallback_image_label(self.url, self.source)
        display_caption = self.caption or self.alt or _fallback_image_label(self.url, self.source)
        normalized_url = _normalize_url(self.url)
        return {
            "url":           normalized_url,
            "image_url":     normalized_url,
            "thumbnail_url": normalized_url,
            "src":           normalized_url,
            "alt":           display_alt,
            "title":         display_alt,
            "display_title": display_alt,
            "source":        self.source,
            "page_url":      self.page_url,
            "score":         round(float(self.score), 3),
            "confidence":    self.confidence,
            "width":         self.width,
            "height":        self.height,
            "caption":       display_caption,
            "scored_by":     self.scored_by,
            "domain":        self.domain,
        }


def _make_candidate(
    *,
    url:         str,
    alt:         str,
    source:      str,
    page_url:    str,
    short_query: str,
    caption:     str = "",
    title:       str = "",
    width:       int = 0,
    height:      int = 0,
    page_text:   str = "",
) -> Optional[CandidateImage]:
    if not url or not _is_valid_image_url(url):
        return None
    if _is_blocked(url, f"{alt} {caption} {title}"):
        return None

    effective_alt = (alt or caption or title or _fallback_image_label(url, source)).strip()
    generic_alt_values = {
        "", "image", "images", "figure", "fig", "photo", "picture",
        "uncaptioned image", "[uncaptioned image]", "untitled image",
        "no caption", "no caption available", "image unavailable",
    }
    if effective_alt.lower() in generic_alt_values:
        effective_alt = _fallback_image_label(url, source)

    if _alt_is_junk(effective_alt, source):
        return None

    # FIX: Early discard — check alt/url patterns before scoring anything
    if _post_score_discard(url, effective_alt):
        return None

    # FIX: For meta_image/page_img on AI/ML queries, require diagram signals
    # This catches the office photo and comic at candidate creation time.
    if _image_is_ai_ml_query(short_query) and _is_clearly_not_diagram(url, effective_alt, source):
        logger.debug(
            f"_make_candidate | rejected (no diagram signal) [{source}]: {effective_alt[:60]}"
        )
        return None

    hits, ratio = _term_overlap(short_query, title, alt, caption, page_text[:400])
    if hits < MIN_TERM_HITS and source not in ("wikipedia", "arxiv"):
        return None

    is_blog = _is_blog_host(url) or _is_blog_host(page_url)

    score = ratio + SOURCE_BOOST.get(source, 0.0)

    if is_blog:
        score -= 0.08  # FIX: increased penalty from 0.05 to 0.08

    blob   = f"{alt} {caption} {title} {url}".lower()
    intent = _intent(short_query)
    if intent == "diagram":
        if any(k in blob for k in ["diagram", "figure", "architecture", "pipeline", "flow", "system", "overview", "comparison"]):
            score += 0.12
        else:
            score -= 0.05

    if width and height:
        if width < 180 or height < 180:
            if not _is_trusted(url):
                score -= 0.12
        elif width >= 600 and height >= 400:
            score += 0.05

    score = round(min(max(score, 0.0), 1.0), 3)

    if score < SOURCE_MIN_SCORE.get(source, MIN_SCORE):
        return None

    confidence = "high" if score >= 0.70 else "medium" if score >= 0.50 else "low"
    return CandidateImage(
        url        = _normalize_url(url),
        alt        = (alt or caption or title or "Image")[:180],
        source     = source,
        page_url   = page_url,
        score      = score,
        confidence = confidence,
        width      = width,
        height     = height,
        caption    = caption[:250],
        scored_by  = "lexical",
        domain     = _domain(url),
        term_hits  = hits,
        is_blog    = is_blog,
    )


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------

async def _wikipedia_titles(short_query: str) -> List[str]:
    q_lower = short_query.lower()
    titles: List[str] = []
    for keyword, article_titles in WIKIPEDIA_TOPIC_MAP.items():
        if keyword in q_lower:
            titles.extend(article_titles)
    titles = list(dict.fromkeys(titles))[:4]
    if titles:
        return titles
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                WIKIPEDIA_SEARCH_URL,
                params={"action": "query", "format": "json", "list": "search",
                        "srsearch": short_query[:80], "srnamespace": "0", "srlimit": "4"},
                headers={"User-Agent": BROWSER_UA},
            )
            if resp.status_code == 200:
                results = resp.json().get("query", {}).get("search", [])
                titles  = [r["title"].replace(" ", "_") for r in results if r.get("title")]
    except Exception as e:
        logger.debug(f"Wikipedia search failed: {e}")
    return titles[:3]


_WIKIPEDIA_JUNK_FILES = {
    "commons-logo", "symbol_category", "symbol_list", "symbol_book",
    "folder_hexagonal", "edit-clear", "ambox", "question_book",
    "wikibooks-logo", "wikiquote-logo", "wikisource-logo",
    "wiki_letter", "text-x-generic", "crystal_clear",
    "nuvola", "gnome-", "gtk-", "fairytale",
}


async def _get_wikipedia_page_images(
    title: str, short_query: str,
) -> List[CandidateImage]:
    out: List[CandidateImage] = []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(
                WIKIPEDIA_SEARCH_URL,
                params={
                    "action": "query", "format": "json",
                    "titles": title.replace("_", " "),
                    "prop": "images", "imlimit": "10",
                },
                headers={"User-Agent": BROWSER_UA},
            )
            if resp.status_code != 200:
                return []
            pages = resp.json().get("query", {}).get("pages", {})
            image_titles: List[str] = []
            for _, page in pages.items():
                for img in page.get("images", []):
                    ft = img.get("title", "")
                    if not ft:
                        continue
                    ft_lower = ft.lower().replace(" ", "_")
                    if any(junk in ft_lower for junk in _WIKIPEDIA_JUNK_FILES):
                        continue
                    if not any(ft_lower.endswith(ext) for ext in (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")):
                        continue
                    image_titles.append(ft)

            if not image_titles:
                logger.debug(f"Wikipedia page-images: no content images for '{title}'")
                return []

            for img_title in image_titles[:5]:
                try:
                    info_resp = await client.get(
                        WIKIPEDIA_SEARCH_URL,
                        params={
                            "action": "query", "format": "json",
                            "titles": img_title,
                            "prop": "imageinfo",
                            "iiprop": "url|size|mime",
                        },
                        headers={"User-Agent": BROWSER_UA},
                    )
                    if info_resp.status_code != 200:
                        continue
                    info_pages = info_resp.json().get("query", {}).get("pages", {})
                    for _, ip in info_pages.items():
                        ii = ip.get("imageinfo", [])
                        if not ii:
                            continue
                        info = ii[0]
                        img_url = info.get("url", "")
                        if not img_url:
                            continue
                        mime = info.get("mime", "")
                        if mime == "image/svg+xml" and int(info.get("size", 0) or 0) < 3000:
                            continue

                        clean_title = re.sub(
                            r"\.(png|jpg|jpeg|svg|gif|webp)$", "",
                            img_title.replace("File:", "").replace("_", " "),
                            flags=re.I,
                        )
                        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"

                        cand = _make_candidate(
                            url=img_url,
                            alt=clean_title[:180],
                            source="wikipedia",
                            page_url=page_url,
                            short_query=short_query,
                            caption=clean_title,
                            title=clean_title,
                            width=int(info.get("width", 0) or 0),
                            height=int(info.get("height", 0) or 0),
                        )
                        if cand:
                            out.append(cand)
                except Exception as e:
                    logger.debug(f"Wikipedia imageinfo failed for '{img_title}': {e}")

    except Exception as e:
        logger.debug(f"Wikipedia page-images failed for '{title}': {e}")

    if out:
        logger.info(
            f"FIX 1A | Wikipedia page-images fallback: {len(out)} images from '{title}'"
        )
    return out


async def _get_wikipedia_images(short_query: str) -> List[CandidateImage]:
    titles = await _wikipedia_titles(short_query)
    if not titles:
        logger.debug(f"Wikipedia images: no article titles for '{short_query}'")
        return []

    seen_titles: Set[str] = set()
    unique_titles: List[str] = []
    for t in titles:
        t_norm = t.lower().replace(" ", "_")
        if t_norm not in seen_titles:
            seen_titles.add(t_norm)
            unique_titles.append(t)
    if len(unique_titles) < len(titles):
        logger.info(
            f"FIX 1D | Deduped Wikipedia titles: {len(titles)} → {len(unique_titles)}"
        )
    titles = unique_titles

    out: List[CandidateImage] = []
    for title in titles:
        try:
            encoded = urllib.parse.quote(title, safe="_()")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    WIKIPEDIA_SUMMARY_URL.format(title=encoded),
                    headers={"User-Agent": BROWSER_UA},
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()

            thumb   = data.get("thumbnail", {}) or {}
            img_url = thumb.get("source", "")
            if not img_url:
                logger.debug(
                    f"Wikipedia image: no thumbnail for '{title}' — trying page-images fallback"
                )
                fallback_imgs = await _get_wikipedia_page_images(title, short_query)
                out.extend(fallback_imgs)
                continue
            img_url = re.sub(r"/\d+px-", "/800px-", img_url)

            page_url   = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            title_text = data.get("title", "") or title
            desc_text  = data.get("description", "") or ""
            extract    = data.get("extract", "") or ""

            cand = _make_candidate(
                url=img_url, alt=f"{title_text} - {desc_text}".strip(" -"),
                source="wikipedia", page_url=page_url, short_query=short_query,
                caption=desc_text, title=title_text,
                width=int(thumb.get("width", 0) or 0),
                height=int(thumb.get("height", 0) or 0),
                page_text=extract[:600],
            )
            if cand:
                out.append(cand)
        except Exception as e:
            logger.debug(f"Wikipedia image failed for {title}: {e}")
    logger.info(f"Wikipedia images: {len(out)} candidates from {len(titles)} articles")
    return out


async def _search_wikimedia_commons(short_query: str, limit: int = 6) -> List[CandidateImage]:
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(
                WIKIMEDIA_SEARCH_URL,
                params={
                    "action": "query", "format": "json",
                    "generator": "search", "gsrsearch": short_query[:80],
                    "gsrnamespace": "6", "gsrlimit": str(limit),
                    "prop": "imageinfo", "iiprop": "url|size|mime",
                },
                headers={"User-Agent": BROWSER_UA},
            )
            if resp.status_code != 200:
                return []
            pages = (resp.json().get("query", {}) or {}).get("pages", {}) or {}

        out: List[CandidateImage] = []
        for _, page in pages.items():
            title     = page.get("title", "")
            imageinfo = page.get("imageinfo", []) or []
            if not imageinfo:
                continue
            info    = imageinfo[0]
            img_url = info.get("url", "")
            mime    = info.get("mime", "")
            if not img_url:
                continue
            if mime == "image/svg+xml" and int(info.get("size", 0) or 0) < 5000:
                continue

            clean_title = re.sub(r"\.(png|jpg|jpeg|svg|gif|webp)$", "",
                                  title.replace("File:", "").replace("_", " "), flags=re.I)
            page_url = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))

            cand = _make_candidate(
                url=img_url, alt=clean_title[:180],
                source="wikimedia", page_url=page_url, short_query=short_query,
                caption=clean_title, title=clean_title,
                width=int(info.get("width", 0) or 0),
                height=int(info.get("height", 0) or 0),
            )
            if cand:
                out.append(cand)
        return out[:limit]
    except Exception as e:
        logger.debug(f"Wikimedia Commons failed: {e}")
        return []


def _extract_img_url(raw_url: str, base_url: str) -> Optional[str]:
    if not raw_url or str(raw_url).startswith("data:"):
        return None
    raw_url = str(raw_url).strip()
    if raw_url.startswith("//"):
        return _normalize_url("https:" + raw_url)
    if raw_url.startswith("/"):
        parsed = urlparse(base_url)
        return _normalize_url(f"{parsed.scheme}://{parsed.netloc}{raw_url}")
    if raw_url.startswith("http"):
        return _normalize_url(raw_url)
    return _normalize_url(urljoin(base_url, raw_url))


async def _get_arxiv_images(short_query: str, source_urls: List[str]) -> List[CandidateImage]:
    arxiv_urls = [u for u in source_urls if "arxiv.org" in (u or "")]
    if not arxiv_urls:
        return []

    html_urls: List[str] = []
    for url in arxiv_urls[:4]:
        try:
            if "/abs/" in url:
                pid = url.split("/abs/")[-1].split("?")[0].split("#")[0]
            elif "/pdf/" in url:
                pid = url.split("/pdf/")[-1].replace(".pdf", "").split("?")[0]
            else:
                continue
            pid = re.sub(r"v\d+$", "", pid)
            html_urls.append(f"https://arxiv.org/html/{pid}")
        except Exception:
            continue

    out: List[CandidateImage] = []
    for html_url in html_urls:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(html_url, headers={"User-Agent": BROWSER_UA})
                if resp.status_code != 200:
                    logger.info(f"ArXiv HTML {html_url} → {resp.status_code} — trying /abs/ fallback")
                    abs_url = html_url.replace("/html/", "/abs/")
                    try:
                        abs_resp = await client.get(abs_url, headers={"User-Agent": BROWSER_UA})
                        if abs_resp.status_code == 200:
                            soup_abs = BeautifulSoup(abs_resp.text, "html.parser")
                            og_tag = (
                                soup_abs.find("meta", attrs={"property": "og:image"})
                                or soup_abs.find("meta", attrs={"name": "twitter:image"})
                            )
                            if og_tag:
                                fb_url = _extract_img_url(og_tag.get("content", ""), abs_url)
                                if fb_url:
                                    title_tag = soup_abs.find("title")
                                    page_title = title_tag.get_text(" ", strip=True)[:200] if title_tag else ""
                                    desc_tag = soup_abs.find("meta", attrs={"name": "description"})
                                    page_desc = (desc_tag.get("content", "") or "")[:200] if desc_tag else ""
                                    cand = _make_candidate(
                                        url=fb_url, alt=page_title or page_desc,
                                        source="arxiv", page_url=abs_url,
                                        short_query=short_query,
                                        caption=page_desc, title=page_title,
                                    )
                                    if cand:
                                        out.append(cand)
                                        logger.info(f"ArXiv /abs/ fallback got og:image for {abs_url}")
                    except Exception:
                        pass
                    continue

            soup = BeautifulSoup(resp.text, "html.parser")
            page_snippet = re.sub(r"\s+", " ", soup.get_text(" ")[:800])

            for figure in soup.find_all("figure")[:10]:
                img_tag = figure.find("img")
                if not img_tag:
                    continue

                img_url = _extract_img_url_from_tag(img_tag, html_url)
                if not img_url or not img_url.startswith("http"):
                    continue

                cap_tag = figure.find("figcaption")
                caption = cap_tag.get_text(" ", strip=True)[:400] if cap_tag else ""
                alt     = (img_tag.get("alt") or "").strip()

                if len(caption) < 10 and len(alt) < 8:
                    continue
                if not alt or len(alt) < 5:
                    alt = caption[:100]

                try:
                    w = int(img_tag.get("width",  "0") or "0")
                    h = int(img_tag.get("height", "0") or "0")
                except Exception:
                    w = h = 0

                cand = _make_candidate(
                    url=img_url, alt=alt, source="arxiv",
                    page_url=html_url, short_query=short_query,
                    caption=caption, title=caption[:80],
                    width=w, height=h, page_text=page_snippet,
                )
                if cand:
                    out.append(cand)

            if not out:
                ltx_imgs = soup.select("img.ltx_graphics, img.ltx_Math")[:8]
                if ltx_imgs:
                    logger.info(
                        f"ArXiv | no <figure> tags — trying {len(ltx_imgs)} "
                        f"ltx_graphics images from {html_url}"
                    )
                for ltx_img in ltx_imgs:
                    img_url = _extract_img_url_from_tag(ltx_img, html_url)
                    if not img_url or not img_url.startswith("http"):
                        continue
                    alt = (ltx_img.get("alt") or "").strip()
                    parent = ltx_img.find_parent(["div", "section", "td"])
                    nearby_text = ""
                    if parent:
                        nearby_text = parent.get_text(" ", strip=True)[:200]
                    if not alt or len(alt) < 5:
                        alt = nearby_text[:100] if nearby_text else "ArXiv figure"
                    try:
                        w = int(ltx_img.get("width",  "0") or "0")
                        h = int(ltx_img.get("height", "0") or "0")
                    except Exception:
                        w = h = 0
                    cand = _make_candidate(
                        url=img_url, alt=alt, source="arxiv",
                        page_url=html_url, short_query=short_query,
                        caption=nearby_text[:200], title=alt[:80],
                        width=w, height=h, page_text=page_snippet,
                    )
                    if cand:
                        out.append(cand)

        except Exception as e:
            logger.debug(f"ArXiv scrape failed for {html_url}: {e}")

    if out:
        logger.info(f"ArXiv images: {len(out)} figure candidates from {len(html_urls)} papers")
    else:
        logger.debug(f"ArXiv images: 0 figure candidates from {len(html_urls)} papers")
    return out


async def _scrape_trusted_figures(short_query: str, source_urls: List[str]) -> List[CandidateImage]:
    trusted = [u for u in source_urls if _is_trusted(u) and "arxiv.org" not in u][:3]
    out: List[CandidateImage] = []
    for url in trusted:
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": BROWSER_UA})
                if resp.status_code != 200:
                    continue
            soup = BeautifulSoup(resp.text, "html.parser")
            page_snippet = re.sub(r"\s+", " ", soup.get_text(" ")[:600])
            title_tag   = soup.find("title")
            page_title  = title_tag.get_text(" ", strip=True)[:200] if title_tag else ""

            for figure in soup.find_all("figure")[:6]:
                img_tag = figure.find("img")
                if not img_tag:
                    continue
                img_url = _extract_img_url_from_tag(img_tag, url)
                if not img_url:
                    continue
                cap_tag = figure.find("figcaption")
                caption = cap_tag.get_text(" ", strip=True)[:300] if cap_tag else ""
                alt     = (img_tag.get("alt") or caption[:80] or "").strip()
                try:
                    w = int(img_tag.get("width",  "0") or "0")
                    h = int(img_tag.get("height", "0") or "0")
                except Exception:
                    w = h = 0
                cand = _make_candidate(
                    url=img_url, alt=alt, source="page_img",
                    page_url=url, short_query=short_query,
                    caption=caption, title=page_title,
                    width=w, height=h, page_text=page_snippet,
                )
                if cand:
                    out.append(cand)
        except Exception as e:
            logger.debug(f"Trusted scrape failed for {url}: {e}")
    return out


async def _scrape_meta_images_limited(short_query: str, source_urls: List[str]) -> List[CandidateImage]:
    """
    FIX: meta_image scraping is now much more restricted:
    1. Completely skip for AI/ML queries — meta images are never research diagrams
    2. Skip all blog hosts AND all known marketing/product domains
    3. Only process URLs from genuinely neutral sources
    """
    # FIX: For AI/ML queries, meta_image scraping produces nothing useful.
    # Every AI/ML blog uses a stock photo or marketing graphic as their OG image.
    # Just skip entirely and let arxiv/wikipedia supply the images.
    if _image_is_ai_ml_query(short_query):
        logger.debug(
            f"meta_image scrape | SKIPPED for AI/ML query '{short_query[:40]}'"
        )
        return []

    non_blog = [
        u for u in source_urls
        if not _is_blog_host(u) and not _is_trusted(u)
        and not any(marketing in _domain(u) for marketing in [
            "oracle.com", "ibm.com", "coursera.org", "linearloop.io",
            "pecollective.com", "digitalapplied.com", "anolytics.ai",
            "stack-ai.com", "coreweave.com", "encord.com",
            "solutelabs.com", "therightsw.com", "protecto.ai",
            "mindbreeze.com",
        ])
    ][:2]
    out: List[CandidateImage] = []
    for url in non_blog:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": BROWSER_UA})
                if resp.status_code != 200:
                    continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title_tag = soup.find("title")
            page_title = title_tag.get_text(" ", strip=True)[:200] if title_tag else ""
            desc_tag   = soup.find("meta", attrs={"name": "description"})
            page_desc  = (desc_tag.get("content", "") or "")[:200] if desc_tag else ""

            for key in ["og:image", "twitter:image"]:
                tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
                if not tag:
                    continue
                img_url = _extract_img_url(tag.get("content", ""), url)
                if not img_url:
                    continue
                cand = _make_candidate(
                    url=img_url, alt=page_desc or page_title,
                    source="meta_image", page_url=url, short_query=short_query,
                    caption=page_desc, title=page_title,
                )
                if cand:
                    out.append(cand)
                    break
        except Exception as e:
            logger.debug(f"Meta image scrape failed for {url}: {e}")
    return out


# ---------------------------------------------------------------------------
# Dedup and caps
# ---------------------------------------------------------------------------

def _dedupe(images: List[CandidateImage]) -> List[CandidateImage]:
    seen_norm_urls: Set[str] = set()
    seen_path_keys: Set[str] = set()
    seen_text_sigs: Set[str] = set()
    out: List[CandidateImage] = []
    for img in images:
        norm = _normalize_url(img.url)
        pkey = _url_path_key(img.url)
        toks = _meaningful_terms(f"{img.alt} {img.caption}")[:14]
        tsig = " ".join(sorted(set(toks))[:10])
        if norm in seen_norm_urls:
            continue
        if pkey and pkey in seen_path_keys:
            continue
        if tsig and len(tsig) > 5 and tsig in seen_text_sigs:
            continue
        seen_norm_urls.add(norm)
        if pkey:
            seen_path_keys.add(pkey)
        if tsig:
            seen_text_sigs.add(tsig)
        out.append(img)
    return out


def _apply_caps(images: List[CandidateImage], is_ai_ml: bool = False) -> List[CandidateImage]:
    """FIX: Pass is_ai_ml to set meta_image cap to 0 for AI/ML queries."""
    source_counts: Counter = Counter()
    host_counts:   Counter = Counter()

    # FIX: For AI/ML queries, meta_image cap is 0 (completely blocked)
    effective_caps = dict(SOURCE_CAP)
    if is_ai_ml:
        effective_caps["meta_image"] = 0
        effective_caps["page_img"]   = 2  # also tighten page_img for AI/ML

    out: List[CandidateImage] = []
    for img in sorted(images, key=lambda x: x.score, reverse=True):
        src  = img.source
        host = img.domain or _domain(img.url)
        if source_counts[src] >= effective_caps.get(src, 2):
            continue
        host_cap = 1 if img.is_blog else (3 if _is_trusted(img.url) else 2)
        if host and host_counts[host] >= host_cap:
            continue
        source_counts[src] += 1
        if host:
            host_counts[host] += 1
        out.append(img)
    return out


# ---------------------------------------------------------------------------
# Gemini text-only reranking
# ---------------------------------------------------------------------------

async def _gemini_text_score(alt: str, caption: str, img_url: str, short_query: str) -> float:
    api_key = getattr(settings, "google_api_key", "")
    if not api_key:
        return 0.5

    description = " | ".join(filter(None, [alt[:120], caption[:120]]))
    if not description.strip():
        return 0.5

    prompt = (
        f"Image metadata:\n"
        f"  Caption/Alt: {description}\n"
        f"  URL: {img_url[:100]}\n\n"
        f"Search query: '{short_query}'\n\n"
        "Is this a relevant RESEARCH DIAGRAM or TECHNICAL FIGURE?\n"
        "REJECT: stock photos, office photos, people at computers, cartoon/comic images, "
        "blog thumbnails, marketing graphics, lifestyle photos.\n"
        "ACCEPT: architecture diagrams, pipeline figures, benchmark charts, technical comparisons, "
        "system diagrams from research papers.\n"
        "Reply with ONE digit only:\n"
        "0 = stock photo / blog thumbnail / cartoon / marketing / people / unrelated\n"
        "1 = weakly related\n"
        "2 = moderately related technical image\n"
        "3 = relevant research diagram or figure\n"
        "4 = directly relevant technical diagram from a paper"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 4, "temperature": 0.0},
    }

    try:
        async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
            resp = await client.post(
                GEMINI_TEXT_URL, params={"key": api_key}, json=payload,
            )
            if resp.status_code == 429:
                return -1.0
            if resp.status_code != 200:
                return 0.5
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            m   = re.search(r"[0-4]", raw)
            return round(int(m.group(0)) / 4.0, 2) if m else 0.5
    except Exception as e:
        logger.debug(f"Gemini text score failed: {e}")
        return 0.5


# ---------------------------------------------------------------------------
# Final URL verification
# ---------------------------------------------------------------------------

async def _verify_image_url(url: str) -> bool:
    if not url:
        return False
    headers = {
        "User-Agent": BROWSER_UA,
        "Range": "bytes=0-0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code not in (200, 206):
                return False
            ctype = (resp.headers.get("content-type") or "").lower()
            if ctype and not ctype.startswith("image/"):
                return False
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public retriever
# ---------------------------------------------------------------------------

class ImageRetriever:
    """
    Research-grade image retriever.

    Source priority (highest → lowest):
      1. ArXiv paper figures  (+0.25 boost) — real paper diagrams with captions
      2. Wikimedia Commons    (+0.20 boost) — CC-licensed technical diagrams
      3. Wikipedia            (+0.18 boost) — encyclopedia images
      4. Trusted page figures (+0.00)       — figures from research sites
      5. Meta OG images       (-0.25 penalty, min_score=0.80, BLOCKED for AI/ML queries)

    FIXES applied in this version:
      - meta_image completely blocked for AI/ML queries (no more stock photos/cartoons)
      - _is_clearly_not_diagram() check at candidate creation time
      - Expanded _DISCARD_ALT_PATTERNS to catch comic/cartoon/lifestyle alt text
      - Tighter SOURCE_MIN_SCORE thresholds
      - Gemini prompt explicitly mentions cartoons/stock photos to reject
      - MIN_GEMINI raised from 0.45 → 0.50
      - _apply_caps() takes is_ai_ml flag to zero out meta_image cap
    """

    def __init__(self, min_width: int = 150, min_height: int = 150, max_images: int = 8):
        self.min_width  = min_width
        self.min_height = min_height
        self.max_images = max_images
        self.use_gemini = bool(getattr(settings, "google_api_key", ""))
        logger.info(
            f"ImageRetriever ready | gemini={'ON' if self.use_gemini else 'OFF'} "
            f"| max={max_images} | min_score={MIN_SCORE} | collect_timeout={COLLECT_TIMEOUT}s"
        )

    def _is_too_vague(self, short_query: str) -> bool:
        terms = _meaningful_terms(short_query)
        if len(terms) < MIN_MEANINGFUL_TERMS:
            return True
        generic_only = {"give", "show", "find", "image", "images", "picture", "pictures", "photo", "photos"}
        return all(t in generic_only for t in terms)

    async def _collect(self, short_query: str, source_urls: List[str]) -> List[CandidateImage]:
        variants = _source_query_variants(short_query)

        wiki_queries    = variants.get("wikipedia", [])[:2]
        commons_queries = variants.get("wikimedia",  [])[:3]

        tasks = [
            _get_arxiv_images(short_query, source_urls),
            *[_search_wikimedia_commons(q, limit=8) for q in commons_queries],
            *[_get_wikipedia_images(q) for q in wiki_queries],
            _scrape_trusted_figures(short_query, source_urls),
            _scrape_meta_images_limited(short_query, source_urls),
        ]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=COLLECT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Image collect timed out after {COLLECT_TIMEOUT}s — partial results")
            results = []

        candidates: List[CandidateImage] = []
        for res in results:
            if isinstance(res, list):
                candidates.extend(res)

        if not candidates:
            expanded = _expand_acronyms(short_query)
            if expanded != short_query:
                logger.info(f"Image fallback | trying expanded query: '{expanded}'")
                try:
                    fallback = await asyncio.wait_for(
                        _search_wikimedia_commons(expanded, limit=6),
                        timeout=10,
                    )
                    candidates.extend(fallback)
                except Exception:
                    pass

        return candidates

    async def _gemini_refine(
        self,
        short_query: str,
        images: List[CandidateImage],
        is_ai_ml: bool = False,
    ) -> List[CandidateImage]:
        if not self.use_gemini or not images:
            return images

        # FIX: For AI/ML queries, run ALL non-arxiv/non-wikimedia images through
        # Gemini — not just borderline ones. We want to catch anything that
        # slipped past the lexical filters.
        if is_ai_ml:
            to_score = [
                img for img in images
                if img.source not in ("arxiv", "wikimedia")
            ]
        else:
            to_score = (
                [img for img in images if img.source == "meta_image"] +
                [img for img in images if img.source not in ("meta_image",) and 0.42 <= img.score <= 0.75][:4]
            )

        updated: Dict[str, float] = {}
        for img in to_score:
            gs = await _gemini_text_score(
                alt=img.alt, caption=img.caption,
                img_url=img.url, short_query=short_query,
            )
            if gs == -1.0:
                break
            updated[img.url] = gs

        final: List[CandidateImage] = []
        for img in images:
            if img.url in updated:
                new_score = updated[img.url]
                if new_score < MIN_GEMINI:
                    logger.info(
                        f"Gemini REJECTED [{img.source}] score={new_score:.2f}: {img.alt[:60]}"
                    )
                    continue
                img.score     = max(img.score, new_score)
                img.scored_by = "gemini_text"
            final.append(img)
        return final

    async def retrieve(self, query: str, source_urls: Optional[List[str]] = None) -> List[Dict]:
        if source_urls is None:
            source_urls = []
        if isinstance(query, list):
            query = " ".join(query)

        query       = _normalize_text(query)
        short_query = _extract_short_query(query)
        is_ai_ml    = _image_is_ai_ml_query(query)

        logger.info(
            f"ImageRetriever | original='{query[:60]}' "
            f"| short='{short_query}' | urls={len(source_urls)}"
        )

        if self._is_too_vague(short_query):
            logger.debug("Query too vague — returning no results")
            return []

        candidates = await self._collect(short_query, source_urls)
        n_arxiv = sum(1 for c in candidates if c.source == "arxiv")
        n_wiki  = sum(1 for c in candidates if c.source in ("wikimedia", "wikipedia"))
        n_meta  = sum(1 for c in candidates if c.source == "meta_image")
        logger.info(f"Raw candidates: {len(candidates)} (arxiv={n_arxiv} wiki={n_wiki} meta={n_meta})")

        if not candidates:
            return []

        candidates = _dedupe(candidates)
        logger.info(f"After dedup: {len(candidates)}")

        # FIX: Pass is_ai_ml to gemini_refine for broader scoring on AI/ML queries
        candidates = await self._gemini_refine(short_query, candidates, is_ai_ml=is_ai_ml)
        logger.info(f"After Gemini: {len(candidates)}")

        # Post-scoring discard filter (belt-and-suspenders)
        pre_discard = len(candidates)
        candidates = [
            img for img in candidates
            if not _post_score_discard(img.url, img.alt)
        ]
        discarded = pre_discard - len(candidates)
        if discarded:
            logger.info(f"FIX 9A | post-scoring discard removed {discarded} images")

        # People/office filter for AI/ML queries
        if is_ai_ml:
            pre_people = len(candidates)
            candidates = [
                img for img in candidates
                if not _DISCARD_ALT_PEOPLE.search(img.alt)
            ]
            dropped_people = pre_people - len(candidates)
            if dropped_people:
                logger.info(f"People/office filter removed {dropped_people} images")

        # AI/ML domain relevance check
        if is_ai_ml:
            pre_domain = len(candidates)
            domain_filtered = []
            for img in candidates:
                alt_lower = img.alt.lower()
                has_ai_term = any(t in alt_lower for t in _IMAGE_AI_ML_TERMS)
                if not has_ai_term and img.source not in ("arxiv", "wikimedia", "wikipedia"):
                    img.score -= 0.15
                    if img.score < SOURCE_MIN_SCORE.get(img.source, MIN_SCORE):
                        logger.debug(
                            f"FIX 9C | dropped image (no AI/ML terms in alt): '{img.alt[:50]}'"
                        )
                        continue
                domain_filtered.append(img)
            candidates = domain_filtered
            domain_dropped = pre_domain - len(candidates)
            if domain_dropped:
                logger.info(f"AI/ML domain check dropped {domain_dropped} images")

        # Per-source min score final filter
        filtered: List[CandidateImage] = []
        for img in candidates:
            if img.score < SOURCE_MIN_SCORE.get(img.source, MIN_SCORE):
                continue
            if img.width and img.width < self.min_width and not _is_trusted(img.url):
                continue
            if img.height and img.height < self.min_height and not _is_trusted(img.url):
                continue
            filtered.append(img)

        filtered.sort(key=lambda x: x.score, reverse=True)
        # FIX: Pass is_ai_ml to _apply_caps so meta_image cap = 0 for AI/ML
        filtered = _apply_caps(filtered, is_ai_ml=is_ai_ml)

        # Verify that the final image URLs are actually fetchable. This prevents
        # broken placeholder cards from reaching the UI.
        final: List[CandidateImage] = []
        for img in filtered:
            if len(final) >= self.max_images:
                break
            if await _verify_image_url(img.url):
                final.append(img)
            else:
                logger.debug(
                    f"Dropped inaccessible image [{img.source}] score={img.score:.2f}: {img.url[:100]}"
                )

        if not final:
            logger.debug("No relevant images found for this query")
            return []

        logger.info(f"Returning {len(final)} images")
        for i, img in enumerate(final, 1):
            logger.info(
                f"  [{i}] score={img.score:.2f} hits={img.term_hits} "
                f"src={img.source} blog={img.is_blog} scored_by={img.scored_by} "
                f"short='{short_query}' alt='{img.alt[:60]}'"
            )

        return [img.to_dict() for img in final]
