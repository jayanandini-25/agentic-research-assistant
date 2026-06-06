"""
core/models.py — Phase 11 Fixed
=====================================================================
File location:  core/models.py

CHANGES vs your existing models.py:
  ResearchResponse gains these new Optional fields (all backward-compatible):
    cited_sources        → Sources tab  (title + url + domain per source)
    rag_chunks           → RAG Chunks tab (flat list with _question field)
    summaries            → Summaries tab
    verification_report  → Verification tab
    verification_status  → Verification tab badge
    deep_research_result → Deep Research tab
    research_tree        → Deep Research tab tree render
    dense_topics_found   → Deep Research tab topic chips
    depth_reached        → Deep Research tab stats

All new fields are Optional so existing code that constructs
ResearchResponse without them will NOT break.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ReportType(str, Enum):
    research   = "research"
    summary    = "summary"
    comparison = "comparison"
    deep       = "deep"


class ReportTone(str, Enum):
    objective  = "objective"
    analytical = "analytical"
    critical   = "critical"
    technical  = "technical"


class ResearchRequest(BaseModel):
    query:          str        = Field(..., min_length=2)
    report_type:    ReportType = Field(ReportType.research)
    tone:           ReportTone = Field(ReportTone.objective)
    use_web:        bool       = Field(True)
    use_local_docs: bool       = Field(False)
    deep_research:  bool       = Field(False)
    max_sources:    int        = Field(10, ge=1, le=30)


class ResearchStatus(str, Enum):
    pending    = "pending"
    planning   = "planning"
    retrieving = "retrieving"
    analyzing  = "analyzing"
    writing    = "writing"
    done       = "done"
    failed     = "failed"


class ImageResult(BaseModel):
    source:     str   = ""
    url:        str   = ""
    alt:        str   = ""
    page_url:   str   = ""
    width:      int   = 0
    height:     int   = 0
    score:      float = 0.0
    caption:    str   = ""
    confidence: str   = "medium"
    scored_by:  str   = "lexical"
    domain:     str   = ""


class SourceDocument(BaseModel):
    source:  str   = ""
    title:   str   = ""
    url:     str   = ""
    content: str   = ""
    score:   float = 0.0


class CitedSource(BaseModel):
    """A source with full metadata for the Sources tab."""
    num:    int  = 0
    title:  str  = ""
    url:    str  = ""
    domain: str  = ""
    source: str  = ""   # "tavily", "arxiv", "wikipedia", etc.


class ResearchResponse(BaseModel):
    session_id: str
    query:      str
    status:     ResearchStatus
    message:    str

    # ── Report tab ─────────────────────────────────────────────────────────
    report:     Optional[str] = None

    # ── Sources (raw URLs — kept for backward compat) ──────────────────────
    sources:    Optional[List[str]] = None

    # ── Sources tab (rich objects: title + domain + url) ───────────────────
    cited_sources: Optional[List[Dict[str, Any]]] = None

    # ── Retrieved docs (by sub-question — used by RAG tab via frontend) ────
    retrieved_docs: Optional[Dict[str, List[dict]]] = None

    # ── RAG Chunks tab (flat list, each chunk has _question field) ──────────
    rag_chunks: Optional[List[Dict[str, Any]]] = None

    # ── Images tab ─────────────────────────────────────────────────────────
    images:     Optional[List[ImageResult]] = None

    # ── Summaries tab ──────────────────────────────────────────────────────
    summaries:  Optional[List[Dict[str, Any]]] = None

    # ── Verification tab ───────────────────────────────────────────────────
    verification_report: Optional[Dict[str, Any]] = None
    verification_status: Optional[str]             = None

    # ── Deep Research tab ──────────────────────────────────────────────────
    deep_research_result: Optional[Dict[str, Any]] = None
    research_tree:        Optional[Dict[str, Any]] = None
    dense_topics_found:   Optional[List[str]]       = None
    depth_reached:        Optional[int]             = None

    # ── Pipeline telemetry (Phase 13) ──────────────────────────────────────
    source_counts:  Optional[Dict[str, int]]       = None
    planner_meta:   Optional[Dict[str, Any]]       = None

    # ── Error ───────────────────────────────────────────────────────────────
    error: Optional[str] = None