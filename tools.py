import os
import logging

from langchain_community.tools.tavily_search import TavilySearchResults
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
    before_sleep_log,
)

from source_ranking import rank_sources, average_quality, format_ranked_results

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Raised when the web search fails after all retry attempts are exhausted.

    Callers MUST catch this explicitly. Unlike the previous version of this
    module, search_web() no longer swallows failures into a string that looks
    like valid research data — a caller that doesn't handle SearchError will
    see the pipeline crash loudly instead of silently generating a report
    from an error message.
    """
    pass


class ConfigError(Exception):
    """Raised for setup problems (e.g. missing API key) that a retry can't fix."""
    pass


def get_search_tool(max_results: int = 5):
    """Returns a Tavily search tool. Free tier = 1000 searches/month.

    Raises ConfigError immediately (no retry) if the API key is missing --
    retrying a request that will never succeed just wastes time and quota.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ConfigError("TAVILY_API_KEY is not set. Retrying will not help.")
    return TavilySearchResults(
        max_results=max_results,
        tavily_api_key=api_key,
    )


def _format_results(results: list) -> str:
    if not results:
        return "No results found."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"{r.get('content', '')}\n"
            f"Source: {r.get('url', '')}"
        )
    return "\n\n".join(formatted)


@retry(
    # Don't waste retries on errors a retry can't fix (bad/missing API key).
    retry=retry_if_not_exception_type(ConfigError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _search_with_retry(query: str, max_results: int) -> list:
    """Single attempt wrapped by tenacity. Raises on failure; tenacity retries
    up to 3 total attempts with exponential backoff (2s, 4s, ... capped at 10s)
    before giving up and re-raising the final exception."""
    tool = get_search_tool(max_results=max_results)
    return tool.invoke(query)


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results.

    Retries transient failures (rate limits, timeouts, connection errors) up
    to 3 times with exponential backoff. If all attempts fail, raises
    SearchError -- it does NOT return an error string that could be mistaken
    for real research data by a downstream agent.
    """
    try:
        results = _search_with_retry(query, max_results)
    except ConfigError:
        raise
    except Exception as e:
        raise SearchError(
            f"Search failed after 3 attempts for query '{query}': {e}"
        ) from e
    return _format_results(results)


def search_web_ranked(query: str, max_results: int = 5) -> dict:
    """Stage 4: same underlying search + retry/backoff behavior as
    search_web() (same exceptions: SearchError, ConfigError), but instead
    of throwing away Tavily's structured per-result data (title, url,
    content, score), this scores and ranks each source deterministically
    -- see source_ranking.py for the full formula and its documented
    limitations.

    search_web() is left completely unchanged and still returns a plain
    string -- single_agent_baseline.py deliberately keeps using it
    unmodified, so the multi-agent vs. single-agent benchmark (Stage 2)
    stays a fair, controlled comparison rather than silently changing one
    side's inputs.

    Returns:
        {
          "formatted": str,   # ranked/annotated text, same shape research_agent
                               # already feeds to the LLM
          "sources": list,    # each: {title, url, content, relevance,
                               # domain_trust, richness, quality_score, quality_tier},
                               # sorted best-first
          "avg_quality": float | None,  # None if there were no sources -- never
                                         # fabricated as 0
        }
    """
    try:
        results = _search_with_retry(query, max_results)
    except ConfigError:
        raise
    except Exception as e:
        raise SearchError(
            f"Search failed after 3 attempts for query '{query}': {e}"
        ) from e

    ranked = rank_sources(results)
    return {
        "formatted": format_ranked_results(ranked),
        "sources": ranked,
        "avg_quality": average_quality(ranked),
    }
