import time
import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.graph import StateGraph, END
from state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools import search_web


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=600,
        api_key=os.environ["GROQ_API_KEY"]
    )


def seq_researcher(state):
    print("\n🔍 Research Agent working...")
    search_results = search_web(state["query"])
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Extract key facts from search results. Be concise."),
        HumanMessage(content=f"Topic: {state['query']}\nResults: {search_results}\nExtract key facts.")
    ])
    print("✅ Research complete.")
    return {"research_data": response.content, "messages": ["researcher done"], "error": ""}


def seq_analyst(state):
    print("\n📊 Analyst Agent working...")
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Analyze these research findings. Be concise."),
        HumanMessage(content=f"Query: {state['query']}\nData: {state['research_data']}\nAnalyze.")
    ])
    print("✅ Analysis complete.")
    return {"analysis": response.content, "messages": ["analyst done"]}


def seq_extra(state):
    print("\n🔎 Extra Search working...")
    search_results = search_web(state["query"] + " statistics 2025")
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Find additional statistics. Bullet points only."),
        HumanMessage(content=f"Topic: {state['query']}\nResults: {search_results}\nFind extra stats.")
    ])
    print("✅ Extra search complete.")
    return {"extra_context": response.content, "messages": ["extra done"]}


def seq_writer(state):
    print("\n✍️  Writer Agent working...")
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Write a structured report in markdown."),
        HumanMessage(content=f"Query: {state['query']}\nResearch: {state['research_data']}\nAnalysis: {state['analysis']}\nExtra: {state['extra_context']}\nWrite report.")
    ])
    print("✅ Report written.")
    return {"report": response.content, "messages": ["writer done"]}


def seq_qa(state):
    print("\n✅ QA Agent checking...")
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Fact-check this report. Output: Verdict, Supported Claims, Unsupported Claims."),
        HumanMessage(content=f"Research: {state['research_data']}\nReport: {state['report']}\nFact-check.")
    ])
    print("✅ QA complete.")
    return {"qa_review": response.content, "messages": ["qa done"]}


def route(state):
    return "end" if state.get("error") else "analyst"


def build_sequential_graph():
    builder = StateGraph(AgentState)
    builder.add_node("researcher", seq_researcher)
    builder.add_node("analyst", seq_analyst)
    builder.add_node("extra_searcher", seq_extra)
    builder.add_node("writer", seq_writer)
    builder.add_node("qa", seq_qa)
    builder.set_entry_point("researcher")
    builder.add_conditional_edges("researcher", route, {"analyst": "analyst", "end": END})
    builder.add_edge("analyst", "extra_searcher")
    builder.add_edge("extra_searcher", "writer")
    builder.add_edge("writer", "qa")
    builder.add_edge("qa", END)
    return builder.compile()


queries = [
    "Impact of social media on mental health",
    "Future of electric vehicles globally"
]

times = []
for i, query in enumerate(queries, 1):
    print(f"\n{'='*50}")
    print(f"Query {i}/{len(queries)}: {query}")
    print('='*50)

    graph = build_sequential_graph()
    start = time.time()
    graph.invoke({
        "query": query, "research_data": "", "analysis": "",
        "extra_context": "", "report": "", "qa_review": "",
        "messages": [], "error": ""
    })
    elapsed = round(time.time() - start, 1)
    times.append(elapsed)
    print(f"\n✅ Time taken: {elapsed}s")

    if i < len(queries):
        print("Waiting 90 seconds...")
        time.sleep(90)

print(f"\n{'='*50}")
print(f"SEQUENTIAL RESULTS")
print(f"{'='*50}")
for i, (q, t) in enumerate(zip(queries, times), 1):
    print(f"Query {i}: {t}s  —  {q}")
print(f"Average: {round(sum(times)/len(times), 1)}s")