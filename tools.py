import os
from langchain_community.tools.tavily_search import TavilySearchResults


def get_search_tool():
    """Returns Tavily search tool. Free tier = 1000 searches/month."""
    return TavilySearchResults(
        max_results=5,
        tavily_api_key=os.environ["TAVILY_API_KEY"]
    )


def search_web(query: str) -> str:
    """Search the web and return formatted results."""
    tool = get_search_tool()
    try:
        results = tool.invoke(query)
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
    except Exception as e:
        return f"Search error: {str(e)}"