"""
Single-agent baseline: the same LLM (agents.get_llm()) and the same search
tool (tools.search_web()) as the multi-agent pipeline, but with ONE LLM call
instead of four (research -> analyst + extra_search -> writer -> qa).

FAIRNESS DESIGN (read before changing anything here):
- Uses the SAME TWO search calls the multi-agent pipeline uses (the primary
  query, plus the same "statistics data examples 2025" enrichment query
  that _extra_searcher_async uses in agents.py). If this baseline only did
  ONE search, any quality difference vs. the multi-agent system could just
  be "it had less raw material," not "decomposition helps" -- that would
  not be a controlled comparison. Giving both conditions the same source
  material isolates the one variable this experiment is actually about:
  does splitting the work across multiple LLM calls/roles help, given
  identical inputs?
- The system prompt below deliberately mirrors agents.writer_agent's
  section structure (Executive Summary / Key Findings / Detailed Analysis
  / Conclusions / Key Takeaways) so a length or structure difference in
  the output isn't just an artifact of asking for a different shape of
  report.
- This module does NOT duplicate get_llm() or search_web() -- it imports
  them from the existing, unmodified agents.py / tools.py, so both
  pipelines are guaranteed to run on the exact same model config
  (openai/gpt-oss-120b, reasoning_effort="low", max_tokens=1024) and the
  exact same search wrapper (including its retry/backoff behavior).
- The multi-agent pipeline makes 5 LLM calls total (research, analyst,
  extra_search, writer, qa) vs. this baseline's 1. That cost/latency gap
  is real and IS one of the things the benchmark is supposed to measure
  honestly -- it is not something to "control away."

This does not implement a QA/fact-check step, on purpose: comparing
"4 LLM calls + a QA pass" against "1 LLM call + no QA pass" would conflate
two differences (decomposition AND fact-checking) into one number. Stage 3
(grounded QA) is where fact-checking gets evaluated properly, on its own.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents import get_llm
from tools import search_web, SearchError, ConfigError

SINGLE_AGENT_SYSTEM_PROMPT = """You are a research assistant who writes complete, professional research reports by yourself, from raw search results, in a single pass.

Write a clear, well-structured report in Markdown format with:
- A clear title (# Title)
- Executive Summary section
- Key Findings section with bullet points
- Detailed Analysis section
- Conclusions section
- Key Takeaways at the end

Use the search results provided. Include specific statistics, numbers, and named examples where the search results contain them. Make it professional and easy to read."""


async def run_single_agent(query: str, config=None) -> dict:
    """Runs the single-agent baseline for one query.

    Returns a dict shaped like a trimmed-down version of the multi-agent
    pipeline's state dict, so the benchmark harness can treat both
    pipelines uniformly: {"query", "report", "error"}.

    `config` is forwarded to llm.ainvoke() so the benchmark harness's
    token-usage callback (attached via config={"callbacks": [...]}) also
    fires for this pipeline, exactly as it does for the multi-agent one --
    required for a fair token/cost comparison, not just a fair latency one.
    """
    try:
        primary_results = search_web(query)
    except (SearchError, ConfigError) as e:
        return {"query": query, "report": "", "error": str(e)}

    try:
        extra_results = search_web(query + " statistics data examples 2025")
    except (SearchError, ConfigError) as e:
        # Same non-critical treatment as agents._extra_searcher_async:
        # don't fail the whole run over the optional enrichment search.
        extra_results = f"[No additional statistics available -- extra search failed: {e}]"

    llm = get_llm()
    messages = [
        SystemMessage(content=SINGLE_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""Topic: {query}

Search results:
{primary_results}

Additional search results:
{extra_results}

Write a comprehensive, professional report using the above.""")
    ]

    response = await llm.ainvoke(messages, config=config)
    return {"query": query, "report": response.content, "error": ""}
