from langgraph.graph import StateGraph, END
from state import AgentState
from agents import (
    research_agent,
    analyst_agent,
    extra_searcher_agent,
    writer_agent,
    qa_agent,
    should_continue
)


def build_graph():
    """
    Parallel pipeline:

    researcher
        ↓         ↘
    analyst    extra_searcher   ← these two run in PARALLEL
        ↓         ↙
       writer
        ↓
        qa
        ↓
       END
    """
    builder = StateGraph(AgentState)

    builder.add_node("researcher", research_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("extra_searcher", extra_searcher_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("qa", qa_agent)

    builder.set_entry_point("researcher")

    # researcher fans out to BOTH analyst and extra_searcher
    builder.add_conditional_edges(
        "researcher",
        should_continue,
        {"analyst": "analyst", "end": END}
    )
    builder.add_edge("researcher", "extra_searcher")

    # both converge at writer
    builder.add_edge("analyst", "writer")
    builder.add_edge("extra_searcher", "writer")

    builder.add_edge("writer", "qa")
    builder.add_edge("qa", END)

    return builder.compile()


def run_research(query: str) -> dict:
    graph = build_graph()
    initial_state = {
        "query": query,
        "research_data": "",
        "analysis": "",
        "extra_context": "",
        "report": "",
        "qa_review": "",
        "messages": [],
        "error": ""
    }
    return graph.invoke(initial_state)