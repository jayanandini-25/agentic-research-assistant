"""
Writer Agent — Phase 12 (Clean References, No Images Section)
=====================================================================
File location:  app/agents/writer_agent.py

CHANGES vs Phase 11:
  1. NO images section — images are completely removed from the report.
     (Frontend handles image display separately.)

  2. References section — completely rewritten for clean rendering:
     - Each reference on ONE line: [N] Title — domain — URL
     - No &nbsp; indentation (caused overflow in the screenshot)
     - No two-line URL split (was breaking layout)
     - URL shortened to domain + path, truncated at 80 chars if needed
     - Grouped: Academic → Encyclopedia → Web

  3. Better report structure:
     - Proper Executive Summary (crisp, 3-5 sentences)
     - Richer analytical body sections (no bullets, pure prose)
     - Smart section assignment with keyword-based fallback
     - Conclusion with clear takeaways

  4. All sections always drafted — sections without assigned summaries
     get a synthesis pass from all summaries.

  5. Longer section output (700 tokens target).

  FIX (this version):
     BUG FIX — All LLM calls now have exponential backoff retry on 429 /
               queue_exceeded errors (up to 4 attempts: 5 → 10 → 20 → 40 s).
               Affects: _draft_section, _draft_synthesis_section,
                        _write_executive_summary, _write_conclusion,
                        _detect_query_type, _plan_sections.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config.settings import get_settings
from core.logger import setup_logger

settings = get_settings()
logger   = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Section templates per query type
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TEMPLATES = {
    "comparison": [
        "Executive Summary",
        "Background & Context",
        "Overview of {entity_a}",
        "Overview of {entity_b}",
        "Head-to-Head Comparison",
        "Performance & Benchmarks",
        "Use Cases & When to Choose Which",
        "Limitations & Trade-offs",
        "Conclusion & Recommendation",
    ],
    "technical": [
        "Executive Summary",
        "Background & Motivation",
        "Core Architecture & Design",
        "Implementation & How It Works",
        "Performance, Evaluation & Results",
        "Limitations & Known Challenges",
        "Recent Advances & Future Directions",
        "Conclusion",
    ],
    "survey": [
        "Executive Summary",
        "Introduction & Scope",
        "Historical Context & Evolution",
        "Current State of the Field",
        "Key Approaches & Methodologies",
        "Comparative Analysis",
        "Open Problems & Future Directions",
        "Conclusion",
    ],
    "general": [
        "Executive Summary",
        "Background & Context",
        "Key Findings",
        "Deep Analysis",
        "Implications & Applications",
        "Limitations & Caveats",
        "Conclusion",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionPlan:
    title         : str
    assigned_sums : List[Dict] = field(default_factory=list)
    order         : int        = 0


@dataclass
class DraftedSection:
    title   : str
    content : str
    order   : int


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit_error(e: Exception) -> bool:
    """Return True for Cerebras / OpenAI 429 / queue_exceeded errors."""
    err_str = str(e).lower()
    return (
        "429" in err_str
        or "queue_exceeded" in err_str
        or "too_many_requests" in err_str
        or "rate limit" in err_str
        or "high traffic" in err_str
    )


async def _llm_with_retry(coro_fn, label: str, fallback, max_attempts: int = 4):
    """
    Call an async coroutine factory `coro_fn()` with exponential backoff on
    rate-limit errors.  Returns the result or `fallback` on exhaustion.

    Usage:
        result = await _llm_with_retry(
            lambda: client.chat.completions.create(...),
            label="WriterAgent | _draft_section 'Foo'",
            fallback="",
        )
    """
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = 5 * (2 ** attempt)   # 5 → 10 → 20 → 40 s
                logger.warning(
                    f"{label} | 429/queue_exceeded attempt {attempt + 1}/{max_attempts} "
                    f"— retrying in {wait}s"
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"{label} | non-retryable error: {e}")
                return fallback
    logger.error(f"{label} | failed after {max_attempts} attempts — using fallback")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Query/section alignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _best_section_index(
    query: str,
    question: str,
    summary: str,
    plans: List["SectionPlan"],
) -> int:
    """Deterministic fallback scorer for section assignment."""
    combined = " ".join([query or "", question or "", summary or ""]).lower()
    combined_tokens = _tokenize_words(combined)
    question_tokens = _tokenize_words(question)
    summary_tokens = _tokenize_words(summary)
    query_tokens = _tokenize_words(query)

    best_idx = 0
    best_score = float("-inf")

    for idx, plan in enumerate(plans):
        title_lower = plan.title.lower()
        title_tokens = _tokenize_words(plan.title)
        score = 0.0

        # direct lexical overlap
        score += 3.0 * len(question_tokens & title_tokens)
        score += 2.0 * len(summary_tokens & title_tokens)
        score += 1.5 * len(query_tokens & title_tokens)

        # section-specific boosts
        if "background" in title_lower or "context" in title_lower or "introduction" in title_lower:
            if any(k in combined for k in ["background", "history", "origin", "context", "motivation", "introduction"]):
                score += 3.0
        if "architecture" in title_lower or "implementation" in title_lower or "mechanism" in title_lower or "design" in title_lower:
            if any(k in combined for k in ["architecture", "implementation", "mechanism", "design", "how it works", "process"]):
                score += 3.5
        if "performance" in title_lower or "benchmark" in title_lower or "results" in title_lower:
            if any(k in combined for k in ["performance", "benchmark", "speed", "latency", "throughput", "accuracy", "evaluation", "result"]):
                score += 4.0
        if "limitations" in title_lower or "trade-off" in title_lower or "tradeoffs" in title_lower or "caveats" in title_lower:
            if any(k in combined for k in ["limitation", "challenge", "weakness", "drawback", "tradeoff", "risk", "constraint", "caveat"]):
                score += 4.0
        if "use cases" in title_lower or "applications" in title_lower or "implications" in title_lower:
            if any(k in combined for k in ["application", "use case", "deployment", "practical", "industry", "real-world", "implication"]):
                score += 3.0
        if "comparison" in title_lower or "head-to-head" in title_lower:
            if any(k in combined for k in ["compare", "comparison", "vs", "versus", "difference", "tradeoff", "contrast"]):
                score += 3.5
        if "conclusion" in title_lower:
            if any(k in combined for k in ["conclusion", "overall", "implication", "takeaway", "recommendation", "future", "open question"]):
                score += 3.5
        if "executive summary" in title_lower:
            score += 0.5 if summary_tokens else 0.0

        score += 0.25 * len(combined_tokens & title_tokens)

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# Writer Agent
# ─────────────────────────────────────────────────────────────────────────────

class WriterAgent:
    """
    Phase 12 Writer Agent.

    Key behaviour:
    - No images section (removed entirely).
    - References on single clean lines — no overflow.
    - All sections always drafted.
    - Richer, longer analytical prose.
    - All LLM calls have 429 retry with exponential backoff.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key  = settings.openai_api_key,
            base_url = settings.openai_base_url or None,
        )
        self.model      = settings.openai_model
        self.fast_model = getattr(settings, "openai_fast_model", settings.openai_model)
        logger.info(f"WriterAgent initialized | model={self.model}")

    # ─────────────────────────────────────────────────────────────────────────
    # Query type detection
    # ─────────────────────────────────────────────────────────────────────────

    async def _detect_query_type(self, query: str) -> str:
        q = query.lower()
        if any(kw in q for kw in ["vs", "versus", "compare", "difference between",
                                    "better", "pros and cons", "which is"]):
            return "comparison"
        if any(kw in q for kw in ["how does", "architecture", "implementation",
                                    "algorithm", "technical", "system design",
                                    "how do", "mechanism"]):
            return "technical"
        if any(kw in q for kw in ["survey", "overview of", "history of",
                                    "evolution of", "state of the art",
                                    "recent advances", "landscape"]):
            return "survey"

        label    = "WriterAgent | _detect_query_type"
        messages = [{"role": "user", "content":
            f'Classify this research query into exactly one type: '
            f'"comparison", "technical", "survey", or "general".\n'
            f'Query: "{query}"\nReply with ONE word only.'}]

        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = messages,
                max_tokens  = 5,
                temperature = 0.0,
            ),
            label    = label,
            fallback = None,
        )
        if resp is not None:
            qtype = (resp.choices[0].message.content or "").strip().lower()
            if qtype in SECTION_TEMPLATES:
                return qtype
        return "general"

    # ─────────────────────────────────────────────────────────────────────────
    # Plan sections
    # ─────────────────────────────────────────────────────────────────────────

    async def _plan_sections(
        self,
        query      : str,
        query_type : str,
        summaries  : List[Dict],
    ) -> List[SectionPlan]:
        template = list(SECTION_TEMPLATES[query_type])

        if query_type == "comparison":
            parts = re.split(
                r"\s+vs\.?\s+|\s+versus\s+|\s+and\s+",
                query, maxsplit=1, flags=re.I,
            )
            entity_a = parts[0].strip() if len(parts) > 1 else "Option A"
            entity_b = parts[1].strip() if len(parts) > 1 else "Option B"
            template = [
                t.replace("{entity_a}", entity_a).replace("{entity_b}", entity_b)
                for t in template
            ]

        plans = [SectionPlan(title=t, order=i) for i, t in enumerate(template)]

        if not summaries:
            return plans

        # LLM-based assignment
        section_titles_str = "\n".join(f"{i}. {p.title}" for i, p in enumerate(plans))
        summaries_str = "\n".join(
            f'{i}. Q: {s.get("question", "")[:100]}\n   A: {s.get("summary", "")[:160]}'
            for i, s in enumerate(summaries)
        )

        assignments_prompt = (
            f"You have these report sections:\n{section_titles_str}\n\n"
            f"For each summary below, reply with ONLY the section number "
            f"(0-{len(plans)-1}) it fits best.\n"
            f"One number per line, in order. No explanation.\n\n"
            f"Summaries:\n{summaries_str}\n\n"
            f"Section assignments (one number per line):"
        )

        assigned_any = False
        assigned_indices = set()
        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": assignments_prompt}],
                max_tokens  = len(summaries) * 5,
                temperature = 0.0,
            ),
            label    = "WriterAgent | _plan_sections",
            fallback = None,
        )

        if resp is not None:
            try:
                lines = (resp.choices[0].message.content or "").strip().splitlines()
                for idx, line in enumerate(lines):
                    if idx >= len(summaries):
                        break
                    try:
                        sec_idx = int(line.strip().split()[0])
                        if 0 <= sec_idx < len(plans):
                            plans[sec_idx].assigned_sums.append(summaries[idx])
                            assigned_any = True
                            assigned_indices.add(idx)
                    except (ValueError, IndexError):
                        pass
            except Exception:
                pass

        # Deterministic fallback for any unassigned summaries.
        # This is more semantic than the old raw keyword-overlap fallback.
        if not assigned_any:
            logger.warning("WriterAgent | section assignment failed — heuristic fallback")

        for idx, s in enumerate(summaries):
            if idx in assigned_indices:
                continue
            best_idx = _best_section_index(
                query=query,
                question=s.get("question", ""),
                summary=s.get("summary", ""),
                plans=plans,
            )
            plans[best_idx].assigned_sums.append(s)

        return plans

    # ─────────────────────────────────────────────────────────────────────────
    # Citation map
    # ─────────────────────────────────────────────────────────────────────────

    def _build_citation_map(
        self,
        summaries : List[Dict],
        all_docs  : List[Dict] = None,
    ) -> Dict[str, int]:
        """
        Only include URLs from RAG-selected summaries.
        
        Previously, secondary docs from all_docs with rag_score >= 0.35 were
        added, but retriever scores are NOT RAG relevance scores — papers about
        OCaml tools, UAV path planning, and CO2-Brine were getting included.
        Now: ONLY URLs that appear in summaries' sources are cited.
        """
        citation_map: Dict[str, int] = {}
        counter = 1
        # ONLY primary: URLs from summaries (RAG-selected)
        for s in summaries:
            for url in s.get("sources", []):
                if url and url not in citation_map:
                    citation_map[url] = counter
                    counter += 1
        logger.info(
            f"WriterAgent | citation map built | {len(citation_map)} refs (primary only)"
        )
        return citation_map

    # ─────────────────────────────────────────────────────────────────────────
    # Draft a body section
    # ─────────────────────────────────────────────────────────────────────────

    async def _draft_section(
        self,
        plan          : SectionPlan,
        query         : str,
        citation_map  : Dict[str, int],
        prev_title    : Optional[str] = None,
        all_summaries : List[Dict]    = None,
    ) -> str:
        if not plan.assigned_sums:
            return ""

        context_parts = []
        for s in plan.assigned_sums:
            sources_str = " ".join(
                f"[{citation_map.get(url, '?')}]"
                for url in s.get("sources", [])[:4]
                if url in citation_map
            )
            text = s.get("summary", "").strip()
            if text:
                context_parts.append(
                    f"Finding (cite as {sources_str or '[?]'}):\n{text}"
                )
        context = "\n\n".join(context_parts)

        transition = (
            f"Open with a smooth one-sentence transition from '{prev_title}'."
            if prev_title and prev_title.lower() != "executive summary"
            else ""
        )

        extra_context = ""
        if all_summaries and len(plan.assigned_sums) < 2:
            snippets = [
                s.get("summary", "")[:150]
                for s in all_summaries
                if s not in plan.assigned_sums and s.get("summary", "")
            ][:3]
            if snippets:
                extra_context = (
                    "\n\nAdditional context (use for synthesis, do not cite directly):\n"
                    + "\n".join(f"- {e}" for e in snippets)
                )

        prompt = f"""You are writing a section of a high-quality academic research report.

Report topic: "{query}"
Section: "{plan.title}"
{transition}

Write 3-5 well-structured paragraphs for this section.

STRICT REQUIREMENTS:
- Every paragraph MUST stay directly relevant to the research query: "{query}"
- If you discuss a subtopic, explicitly connect it back to "{query}"
- Academic prose only — absolutely NO bullet points or numbered lists
- Each paragraph makes a clear analytical point
- Place citation numbers inline naturally: "Studies show X [1], while Y [2] suggests..."
- Do NOT write the section heading
- Be specific, analytical, and insightful
- Minimum 250 words, target 350-450 words
- Do NOT invent or fabricate any claims not supported by the findings below
- If a finding is uncertain, say so explicitly
- Do NOT drift into general background unless it directly supports "{query}"

FACTUAL GUARDRAILS (do NOT contradict these):
- RAG and fine-tuning are complementary approaches, not replacements for each other
- RAG reduces but does not eliminate hallucinations
- Transformers were introduced in 2017 ("Attention Is All You Need")
- BERT is a masked/discriminative language model, not generative
- Diffusion models and GANs are fundamentally different architectures

Primary findings (with citation numbers):
{context}
{extra_context}

Write the section body now:"""

        fallback = "\n\n".join(s.get("summary", "") for s in plan.assigned_sums)

        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 700,
                temperature = 0.25,
            ),
            label    = f"WriterAgent | _draft_section '{plan.title}'",
            fallback = None,
        )
        if resp is not None:
            return (resp.choices[0].message.content or "").strip()
        return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # Draft synthesis section (no assigned summaries)
    # ─────────────────────────────────────────────────────────────────────────

    async def _draft_synthesis_section(
        self,
        section_title : str,
        query         : str,
        all_summaries : List[Dict],
        citation_map  : Dict[str, int],
        prev_title    : Optional[str] = None,
    ) -> str:
        condensed = []
        for s in all_summaries[:6]:
            text = s.get("summary", "").strip()
            q    = s.get("question", "").strip()
            srcs = " ".join(
                f"[{citation_map.get(url, '?')}]"
                for url in s.get("sources", [])[:2]
                if url in citation_map
            )
            if text:
                condensed.append(f"On '{q}' {srcs}:\n{text[:300]}")

        transition = (
            f"Open with a smooth one-sentence transition from '{prev_title}'."
            if prev_title else ""
        )

        prompt = f"""You are writing a section of a high-quality academic research report.

Report topic: "{query}"
Section: "{section_title}"
{transition}

Using the research context below, write 3-4 focused paragraphs about
"{section_title}" as it relates to "{query}".

STRICT REQUIREMENTS:
- Every paragraph MUST clearly connect back to "{query}"
- Academic prose only — NO bullet points or numbered lists
- Be analytical — make clear arguments and draw connections
- Do NOT write the section heading — body only
- Minimum 200 words, target 280-380 words
- Use citation numbers [N] where the context references them
- Do NOT add generic filler; every claim should support the research topic

Research context:
{chr(10).join(condensed)}

Write the section body for "{section_title}" now:"""

        fallback = f"This section examines {section_title} in the context of {query}."

        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 600,
                temperature = 0.3,
            ),
            label    = f"WriterAgent | _draft_synthesis_section '{section_title}'",
            fallback = None,
        )
        if resp is not None:
            return (resp.choices[0].message.content or "").strip()
        return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # Executive Summary
    # ─────────────────────────────────────────────────────────────────────────

    async def _write_executive_summary(
        self,
        query      : str,
        summaries  : List[Dict],
        query_type : str,
    ) -> str:
        top_findings = "\n".join(
            f"- {s.get('summary', '')[:200]}"
            for s in summaries[:5]
            if s.get("summary", "")
        )

        prompt = f"""Write an executive summary for a research report on:
"{query}"

Key findings:
{top_findings}

Requirements:
- 3-5 sentences maximum
- State what was researched, 2-3 most important findings, and the key implication
- Every sentence must directly answer or clarify "{query}"
- Do NOT start with "This report" or "In this report"
- Be direct and crisp — every sentence must add value
- No bullet points, no headings — pure prose
- Avoid generic filler such as "the topic is important" unless tied to "{query}"

Write the executive summary now:"""

        fallback = (
            f"This research examines {query}, synthesising findings across multiple sources. "
            f"The analysis covers {len(summaries)} research threads and presents "
            f"key insights on the topic."
        )

        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 180,
                temperature = 0.2,
            ),
            label    = "WriterAgent | _write_executive_summary",
            fallback = None,
        )
        if resp is not None:
            return (resp.choices[0].message.content or "").strip()
        return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # Conclusion
    # ─────────────────────────────────────────────────────────────────────────

    async def _write_conclusion(
        self,
        query      : str,
        sections   : List[DraftedSection],
        query_type : str,
    ) -> str:
        overview = "\n".join(
            f"- {s.title}: {s.content[:180]}..."
            for s in sections
            if s.content and s.title.lower() != "executive summary"
        )

        rec_instruction = (
            "End with a clear, specific recommendation: which option to choose and when."
            if query_type == "comparison"
            else "End with a forward-looking statement about where the field is headed."
        )

        prompt = f"""Write the conclusion for a research report on: "{query}"

Sections covered:
{overview}

Write 3-4 strong conclusion paragraphs that:
1. Synthesise the 3 most important insights (don't just re-list sections)
2. Explain what these findings mean together — what's the big picture?
3. {rec_instruction}
4. Name 1-2 open questions or limitations that remain

Requirements:
- Do NOT start with "In conclusion", "To summarize", or "In summary"
- Every paragraph must tie back to "{query}"
- Academic prose, no bullets
- Target 200-280 words
- Avoid generic closing language unless it is specific to "{query}"

Write the conclusion now:"""

        fallback = (
            f"This research comprehensively examined {query}. "
            f"The findings reveal important patterns and areas for future investigation."
        )

        resp = await _llm_with_retry(
            lambda: self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 500,
                temperature = 0.3,
            ),
            label    = "WriterAgent | _write_conclusion",
            fallback = None,
        )
        if resp is not None:
            return (resp.choices[0].message.content or "").strip()
        return fallback

    # ─────────────────────────────────────────────────────────────────────────
    # References  — clean, single-line format, no overflow
    # ─────────────────────────────────────────────────────────────────────────

    def _build_references_section(
        self,
        citation_map : Dict[str, int],
        all_docs     : List[Dict] = None,
    ) -> str:
        """
        Each reference on ONE clean line:
            [N] Title — domain (via source)  URL

        Rules that prevent overflow:
        - No &nbsp; indentation
        - No two-line URL split
        - URL truncated to 80 chars max with ellipsis
        - Title truncated to 90 chars max
        """
        if not citation_map:
            return ""

        # Build metadata per URL from docs
        url_meta: Dict[str, Dict] = {}
        for doc in (all_docs or []):
            url = doc.get("url", "")
            if not url:
                continue
            if url not in url_meta:
                url_meta[url] = {
                    "title" : (doc.get("title",  "") or "").strip(),
                    "source": (doc.get("source", "") or "").strip(),
                }

        def _domain(url: str) -> str:
            m = re.search(r"https?://(?:www\.)?([^/]+)", url)
            return m.group(1) if m else url[:30]

        def _short_url(url: str, max_len: int = 80) -> str:
            if len(url) <= max_len:
                return url
            return url[:max_len - 1] + "…"

        def _classify(url: str, source: str) -> str:
            u = url.lower()
            s = source.lower()
            if any(x in u for x in ["arxiv.org", "pubmed", "doi.org",
                                      "semanticscholar", "researchgate", "ssrn"]):
                return "academic"
            if any(x in s for x in ["arxiv", "pubmed", "semantic_scholar",
                                      "academic", "nature", "springer"]):
                return "academic"
            if any(x in u for x in ["wikipedia.org", "britannica.com"]):
                return "encyclopedia"
            return "web"

        sorted_refs = sorted(citation_map.items(), key=lambda x: x[1])

        groups: Dict[str, List] = {"academic": [], "encyclopedia": [], "web": []}
        for url, num in sorted_refs:
            meta   = url_meta.get(url, {})
            title  = meta.get("title",  "")
            source = meta.get("source", "")
            domain = _domain(url)
            cat    = _classify(url, source)

            title_display = (title[:90] + "…") if len(title) > 90 else title
            if not title_display:
                title_display = domain

            groups[cat].append({
                "num"   : num,
                "title" : title_display,
                "domain": domain,
                "source": source,
                "url"   : _short_url(url),
            })

        group_labels = {
            "academic"    : "📚 Academic & Research Sources",
            "encyclopedia": "📖 Encyclopedia & Reference",
            "web"         : "🌐 Web Sources",
        }

        lines = ["\n---\n", "## References\n"]

        for cat, label in group_labels.items():
            refs = groups[cat]
            if not refs:
                continue
            lines.append(f"### {label}\n")
            for ref in refs:
                src_tag = f" (via {ref['source']})" if ref["source"] else ""
                # Single line per reference — no wrapping, no &nbsp;
                lines.append(
                    f"[{ref['num']}] **{ref['title']}** — "
                    f"`{ref['domain']}`{src_tag} — "
                    f"{ref['url']}"
                )
            lines.append("")  # blank line between groups

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def write(
        self,
        query     : str,
        summaries : List[Dict],
        images    : List[Dict] = None,   # kept in signature for compatibility
        tone      : str        = "analytical",
        all_docs  : List[Dict] = None,
    ) -> str:
        """
        Returns a complete markdown research report:
          - Executive Summary
          - Body sections (prose only)
          - Conclusion
          - References (clean single-line format)

        Images are NOT included — frontend handles image display.
        """
        all_docs = all_docs or []

        logger.info(
            f"WriterAgent | query='{query[:60]}' | "
            f"summaries={len(summaries)} | all_docs={len(all_docs)}"
        )

        # 1. Query type
        query_type = await self._detect_query_type(query)
        logger.info(f"WriterAgent | query_type={query_type}")

        # 2. Citation map (FIX 8C: filtered to RAG-selected docs only)
        citation_map = self._build_citation_map(summaries, all_docs)
        logger.info(f"WriterAgent | citations={len(citation_map)}")

        # 3. Plan sections
        plans = await self._plan_sections(query, query_type, summaries)
        logger.info(f"WriterAgent | sections planned={len(plans)}")

        # FIX 8A: For sections with no assigned summaries, find the most
        # semantically similar summary instead of calling synthesis draft.
        for plan in plans:
            if plan.title.lower() in ("executive summary", "conclusion", "conclusion & recommendation"):
                continue
            if not plan.assigned_sums and summaries:
                # Find the most semantically similar summary via keyword overlap
                section_words = set(w.lower() for w in plan.title.split() if len(w) > 2)
                best_match = None
                best_score = -1
                for s in summaries:
                    q_text = (s.get("question", "") + " " + s.get("summary", "")).lower()
                    q_words = set(w for w in q_text.split() if len(w) > 2)
                    overlap = len(section_words & q_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_match = s
                if best_match:
                    plan.assigned_sums = [best_match]
                    logger.info(
                        f"WriterAgent | FIX 8A | '{plan.title}' had no summaries — "
                        f"assigned nearest match: '{best_match.get('question', '')[:60]}'"
                    )

        # FIX 8E: Cross-section dedup tracker
        previously_stated_claims: List[str] = []

        # 4. Draft all sections
        drafted: List[DraftedSection] = []
        prev_title: Optional[str]     = None

        for plan in plans:
            if plan.title.lower() == "executive summary":
                content = await self._write_executive_summary(query, summaries, query_type)
                drafted.append(DraftedSection(title=plan.title, content=content, order=plan.order))
                prev_title = plan.title
                continue

            if "conclusion" in plan.title.lower():
                prev_title = plan.title
                continue

            if plan.assigned_sums:
                content = await self._draft_section(
                    plan=plan,
                    query=query,
                    citation_map=citation_map,
                    prev_title=prev_title,
                    all_summaries=summaries,
                )
            else:
                # FIX 8A: This branch should now be rare (only if no summaries at all)
                logger.warning(f"WriterAgent | '{plan.title}' still has no summaries after assignment")
                content = f"This section examines {plan.title} as it relates to {query}."

            if not content:
                content = f"This section examines {plan.title} as it relates to {query}."

            # FIX 8E: Cross-section dedup — strip sentences already stated
            if previously_stated_claims and content:
                for claim in previously_stated_claims:
                    # Simple substring check for exact claim duplication
                    if claim in content and len(claim) > 40:
                        content = content.replace(claim, "", 1)
                        logger.debug(
                            f"WriterAgent | dedup | removed repeated claim from '{plan.title}': "
                            f"'{claim[:60]}'"
                        )

            # Track claims from this section for future dedup
            if content:
                sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 40]
                previously_stated_claims.extend(sentences[:5])  # Track up to 5 key sentences

            drafted.append(DraftedSection(title=plan.title, content=content, order=plan.order))
            prev_title = plan.title

        logger.info(f"WriterAgent | body sections drafted={len(drafted)}")

        # 5. Conclusion
        conclusion         = await self._write_conclusion(query, drafted, query_type)
        conclusion_title   = next(
            (p.title for p in plans if "conclusion" in p.title.lower()),
            "Conclusion",
        )

        # 6. References
        refs_section = self._build_references_section(citation_map, all_docs)

        # 7. Count distinct source domains for subtitle
        source_domains = set()
        for doc in all_docs:
            url = doc.get("url", "")
            m = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if m:
                source_domains.add(m.group(1))

        # 8. Assemble report
        parts: List[str] = []

        parts.append(f"# {query}\n")
        # FIX 8D: Correct subtitle: "X documents · Y sources"
        parts.append(
            f"*Research Report · {len(all_docs)} documents · "
            f"{len(source_domains)} sources · "
            f"{len(citation_map)} citations*\n"
        )
        parts.append("---\n")

        for sec in drafted:
            parts.append(f"## {sec.title}\n\n{sec.content}\n")

        parts.append(f"## {conclusion_title}\n\n{conclusion}\n")

        if refs_section:
            parts.append(refs_section)

        full_report = "\n".join(parts)

        logger.info(
            f"WriterAgent | DONE | chars={len(full_report)} | "
            f"sections={len(drafted)+1} | citations={len(citation_map)}"
        )

        return full_report