from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    query: str
    research_data: str
    analysis: str
    extra_context: str
    report: str
    qa_review: str
    grounding_report: str
    grounding_score: float  # None when there were no checkable claims (see grounding.py) -- LangGraph's TypedDict schema doesn't enforce this at runtime, but the value can legitimately be None
    # Stage 4 (source_ranking.py): populated by research_agent (raw, primary-
    # search sources) and combined with the extra-search sources inside
    # parallel_step. Declared here because LangGraph silently drops any
    # state key that isn't part of this TypedDict -- verified empirically in
    # Stage 3 (see test_grounding.py) and confirmed again for these fields
    # in test_source_ranking_integration.py.
    research_sources: list
    source_quality_report: str
    avg_source_quality: float  # None when there were no sources at all -- never fabricated as 0
    messages: Annotated[List[str], operator.add]
    error: str