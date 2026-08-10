import os
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from tools import search_web


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=600,
        api_key=os.environ["GROQ_API_KEY"]
    )


async def research_agent(state: AgentState) -> AgentState:
    print("\n🔍 Research Agent working...")
    query = state["query"]
    search_results = search_web(query)
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a research assistant.
Read the raw search results and extract the most important, accurate facts.
Be thorough. Include statistics, key points, and important details.
Format your output clearly with bullet points."""),
        HumanMessage(content=f"""Topic: {query}

Raw search results:
{search_results}

Extract and summarize the key information about this topic.""")
    ]
    response = await llm.ainvoke(messages)
    print("✅ Research complete.")
    return {
        "research_data": response.content,
        "messages": [f"Research Agent: Gathered data on '{query}'"],
        "error": ""
    }


async def _analyst_async(research_data: str, query: str, llm) -> str:
    messages = [
        SystemMessage(content="""You are a senior analyst and domain expert.
Analyze the research findings and produce deep insights.
- Identify key trends and patterns
- Point out what is most significant
- Highlight contradictions or open debates
- Suggest implications and conclusions
Be analytical and critical."""),
        HumanMessage(content=f"""Query: {query}

Research Data:
{research_data}

Provide a detailed analysis of this information.""")
    ]
    response = await llm.ainvoke(messages)
    return response.content


async def _extra_searcher_async(research_data: str, query: str, llm) -> str:
    enriched_query = query + " statistics data examples 2025"
    search_results = search_web(enriched_query)
    messages = [
        SystemMessage(content="""You are a data researcher.
Find additional statistics, numbers, recent data points, and specific examples.
Focus only on: specific numbers, percentages, dates, and named real-world examples.
Keep it concise - bullet points only."""),
        HumanMessage(content=f"""Main research topic: {query}

Already gathered research:
{research_data[:600]}

Additional search results:
{search_results}

Extract only NEW statistics and specific examples not already covered.""")
    ]
    response = await llm.ainvoke(messages)
    return response.content


async def parallel_step(state: AgentState) -> AgentState:
    print("\n📊 Analyst + 🔎 Extra Search running in PARALLEL...")
    llm = get_llm()
    analysis, extra_context = await asyncio.gather(
        _analyst_async(state["research_data"], state["query"], llm),
        _extra_searcher_async(state["research_data"], state["query"], llm)
    )
    print("✅ Both parallel agents complete.")
    return {
        "analysis": analysis,
        "extra_context": extra_context,
        "messages": ["Analyst Agent + Extra Search Agent: Completed in parallel"]
    }


async def writer_agent(state: AgentState) -> AgentState:
    print("\n✍️  Writer Agent working...")
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a professional report writer.
Write a clear, well-structured report in Markdown format with:
- A clear title (# Title)
- Executive Summary section
- Key Findings section with bullet points
- Detailed Analysis section
- Conclusions section
- Key Takeaways at the end
Make it professional and easy to read."""),
        HumanMessage(content=f"""Query: {state['query']}

Research Data:
{state['research_data']}

Analysis:
{state['analysis']}

Additional Statistics and Examples:
{state['extra_context']}

Write a comprehensive, professional report using ALL of the above.""")
    ]
    response = await llm.ainvoke(messages)
    print("✅ Report written.")
    return {
        "report": response.content,
        "messages": ["Writer Agent: Final report generated"]
    }


async def qa_agent(state: AgentState) -> AgentState:
    print("\n✅ QA Agent checking report...")
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a strict fact-checker.
Verify the report against its original source data.

Output in this exact format:

**Overall Verdict:** [PASS / PASS WITH WARNINGS / FAIL]

**Supported Claims:**
- List claims clearly backed by the research data

**Unsupported or Weak Claims:**
- List claims that cannot be verified from the research data

**Suggested Improvements:**
- List specific improvements

Be strict. If a claim is not in the research data, flag it."""),
        HumanMessage(content=f"""Original Research Data:
{state['research_data']}

Additional Context:
{state['extra_context']}

Report to Fact-Check:
{state['report']}

Fact-check this report against all source data.""")
    ]
    response = await llm.ainvoke(messages)
    print("✅ QA check complete.")
    return {
        "qa_review": response.content,
        "messages": ["QA Agent: Fact-check complete"]
    }


def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "parallel_step"