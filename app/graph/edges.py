"""
LangGraph Edges — Phase 12
=====================================================================
File location:  app/graph/edges.py

CHANGES vs Phase 11:
  - after_verifier: re-retrieval loop REMOVED. verifier_node (FIX 2 in
    nodes.py) now always returns status=WRITING, so the old
    `approved_with_flags + iterations < 2 → retriever_node` branch
    was dead code that would never fire — but left the graph definition
    confused. Cleaned up: after_verifier always goes to depth_judge_node.

  - after_start: always planner_node (unchanged).
  - after_judge: routes sufficient → report_node, needs_deep → deep_research_node.
  - after_deep_research: always report_node.
  - after_report: always end.
  - All other edges unchanged.
"""

from app.graph.state import PipelineStatus, ResearchState


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def after_start(state: ResearchState) -> str:
    return "planner_node"


# ─────────────────────────────────────────────────────────────────────────────
# Standard pipeline edges
# ─────────────────────────────────────────────────────────────────────────────

def after_planner(state: ResearchState) -> str:
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    if not state.get("sub_questions"):
        return "end"
    return "retriever_node"


def after_retriever(state: ResearchState) -> str:
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    if not state.get("all_docs"):
        return "end"
    return "rag_node"


def after_rag(state: ResearchState) -> str:
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    if not state.get("filtered_map"):
        return "end"
    return "summarizer_node"


def after_summarizer(state: ResearchState) -> str:
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    return "verifier_node"


def after_verifier(state: ResearchState) -> str:
    """
    verifier_node always returns status=WRITING (FIX 2 in nodes.py).
    Re-retrieval is handled INSIDE the verifier itself for flagged questions.
    This edge simply passes control to the depth judge.

    The old `approved_with_flags + iterations < 2 → retriever_node` branch
    is removed — it was dead code since verifier_node never sets
    status=RETRIEVING anymore, so the graph would never have taken that path.
    """
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    return "depth_judge_node"


# ─────────────────────────────────────────────────────────────────────────────
# Depth judge routing
# ─────────────────────────────────────────────────────────────────────────────

def after_judge(state: ResearchState) -> str:
    if state.get("status") == PipelineStatus.FAILED:
        return "end"
    decision = state.get("depth_judge_decision", "sufficient")
    if decision == "needs_deep_research":
        return "deep_research_node"
    return "report_node"


# ─────────────────────────────────────────────────────────────────────────────
# Deep research path
# ─────────────────────────────────────────────────────────────────────────────

def after_deep_research(state: ResearchState) -> str:
    # Always go to report — deep_research_node handles its own failures
    # gracefully and sets status=WRITING either way.
    return "report_node"


def after_report(state: ResearchState) -> str:
    return "end"