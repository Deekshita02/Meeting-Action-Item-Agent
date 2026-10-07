"""
Fast variant: extractor -> grounding_check (rule-based, free)
                 -> END if clean, else -> reviewer (LLM) -> END or back to extractor
"""
import re
import time
from langgraph.graph import StateGraph, START, END

from graph_agent import extractor, reviewer, GraphState, MAX_REVISIONS
from agent import RunResult

INJECTION_WORDS = re.compile(
    r"ignore|disregard|instruction|system prompt|previous prompt|override", re.I
)


def _norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def grounding_check(state: GraphState) -> dict:
    """Cheap deterministic check. approved=True means 'looks grounded, skip LLM review'."""
    result = state["result"]
    if result is None or not result.success:
        return {"approved": False, "issues": []}

    transcript = _norm(state["transcript"])
    suspicious = bool(INJECTION_WORDS.search(state["transcript"]))
    for item in result.action_items:
        owner, deadline = item.get("owner"), item.get("deadline")
        if owner and _norm(owner) not in transcript:
            suspicious = True
        if deadline and _norm(deadline) not in transcript:
            suspicious = True
    return {"approved": not suspicious, "issues": []}


def route_after_grounding(state: GraphState) -> str:
    result = state["result"]
    if result is None or not result.success or state["approved"]:
        return END
    return "reviewer"


def route_after_review(state: GraphState) -> str:
    result = state["result"]
    if result is None or not result.success or state["approved"]:
        return END
    if state["iteration"] > MAX_REVISIONS:
        return END
    return "extractor"


def build_fast_graph():
    g = StateGraph(GraphState)
    g.add_node("extractor", extractor)
    g.add_node("grounding_check", grounding_check)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "extractor")
    g.add_edge("extractor", "grounding_check")
    g.add_conditional_edges("grounding_check", route_after_grounding, ["reviewer", END])
    g.add_conditional_edges("reviewer", route_after_review, ["extractor", END])
    return g.compile()


fast_app = build_fast_graph()


def extract_with_review_fast(transcript: str) -> RunResult:
    start = time.time()
    final = fast_app.invoke({
        "transcript": transcript, "result": None, "issues": [],
        "approved": False, "iteration": 0, "review_error": None,
    })
    result = final["result"]
    result.latency_seconds = time.time() - start
    result.review_rounds = final["iteration"]
    if result.success and not final["approved"] and final["issues"]:
        result.unclear_mentions = list(result.unclear_mentions) + [
            f"Reviewer flagged: {i}" for i in final["issues"]
        ]
    return result
