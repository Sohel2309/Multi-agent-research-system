from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    query: str
    research_data: str
    analysis: str
    extra_context: str
    report: str
    qa_review: str
    messages: Annotated[List[str], operator.add]
    error: str