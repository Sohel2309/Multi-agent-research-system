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
    messages: Annotated[List[str], operator.add]
    error: str