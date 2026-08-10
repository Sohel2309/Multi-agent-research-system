import asyncio
import sys
from langgraph.graph import StateGraph, END
from state import AgentState
from agents import research_agent, parallel_step, writer_agent, qa_agent, should_continue


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("researcher", research_agent)
    builder.add_node("parallel_step", parallel_step)
    builder.add_node("writer", writer_agent)
    builder.add_node("qa", qa_agent)

    builder.set_entry_point("researcher")
    builder.add_conditional_edges(
        "researcher",
        should_continue,
        {"parallel_step": "parallel_step", "end": END}
    )
    builder.add_edge("parallel_step", "writer")
    builder.add_edge("writer", "qa")
    builder.add_edge("qa", END)

    return builder.compile()

def run_research(query: str) -> dict:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )

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

    return asyncio.run(graph.ainvoke(initial_state))
   