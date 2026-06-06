"""
LangGraph Pipeline State — Phase 11 + Auto Deep Research
=====================================================================
File location:  app/graph/state.py

Change vs Phase 11:
  - Added `depth_judge_decision` field: the LLM verdict from depth_judge_node.
    Values: "sufficient" | "needs_deep_research"
  - `is_deep_research` is NO LONGER set by routes.py from query keywords.
    It is set by depth_judge_node after reading summaries.
  - All other fields unchanged.
"""

from enum import Enum
from typing import TypedDict, List, Dict, Optional, Any


class PipelineStatus(str, Enum):
    PLANNING      = "planning"
    RETRIEVING    = "retrieving"
    RAG           = "rag_filtering"
    SUMMARIZING   = "summarizing"
    VERIFYING     = "verifying"
    JUDGING       = "judging"        # new: depth_judge_node is running
    WRITING       = "writing"        # report_node (Phase 11)
    DEEP_RESEARCH = "deep_research"  # deep_research_node (Phase 10)
    DONE          = "done"
    FAILED        = "failed"


class ResearchState(TypedDict, total=False):
    """
    Single shared state object passed between all LangGraph nodes.

    total=False → every field is Optional at the TypedDict level.
    Nodes only return the keys they actually update.

    ── Standard pipeline fields (Phases 1–9) ───────────────────────────────
    """

    # ── Input ────────────────────────────────────────────────────────────────
    research_query         : str
    session_id             : str

    # ── Planner output ────────────────────────────────────────────────────────
    sub_questions          : List[str]

    # ── Retriever output ──────────────────────────────────────────────────────
    docs_by_question       : Dict[str, List[Dict]]
    all_docs               : List[Dict]
    all_images             : List[Dict]
    source_counts          : Dict[str, int]

    # ── RAG output ────────────────────────────────────────────────────────────
    filtered_map           : Dict[str, List[Dict]]

    # ── Summarizer output ─────────────────────────────────────────────────────
    summaries              : List[Dict]

    # ── Verifier output ───────────────────────────────────────────────────────
    verified_summaries     : List[Dict]
    verification_report    : Dict
    verification_status    : str

    # ── Depth judge output (auto deep research) ───────────────────────────────
    # "sufficient" → go straight to report_node
    # "needs_deep_research" → run deep_research_node first
    depth_judge_decision   : str

    # ── Report writer output ──────────────────────────────────────────────────
    final_report           : str
    sources                : List[str]

    # ── Pipeline control ──────────────────────────────────────────────────────
    status                 : PipelineStatus
    errors                 : List[str]
    iteration_count        : int

    # ── Deep Research fields ──────────────────────────────────────────────────
    # is_deep_research is now set by depth_judge_node, not by routes.py
    is_deep_research       : bool
    deep_research_depth    : int
    deep_research_breadth  : int
    deep_research_result   : Dict
    research_tree          : Dict
    dense_topics_found     : List[str]
    depth_reached          : int

    # ── Planner metadata (Phase 13) ───────────────────────────────────────────
    # Stores the planner's own dimension analysis so depth_judge can use
    # the SAME dimension set instead of independently extracting via LLM.
    planner_meta           : Dict

    # ── Coverage analysis (Phase 12) ─────────────────────────────────────────
    # Set by depth_judge_node, read by report_node and deep_research_node
    coverage_gaps          : List[str]
    coverage_ratio         : float

    # ── Eval & metrics (Phase 12) ─────────────────────────────────────────────
    timing                 : Dict[str, float]
    token_counts           : Dict[str, Dict[str, int]]