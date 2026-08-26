"""
Stage 4: Deterministic source-quality ranking.

WHY THIS EXISTS (read before changing anything here):
tools.search_web() throws away everything except a flattened string as
soon as Tavily returns results. That means every source gets treated as
equally trustworthy and equally relevant, in whatever order Tavily
happened to return them. Two concrete, demonstrable problems follow from
that:
  1. _extra_searcher_async() (agents.py) truncates research_data to the
     first 600 characters before handing it to the LLM. If a weak/
     irrelevant source happens to be listed first, it can push a strong,
     on-topic source out of that window entirely -- the LLM never sees it.
  2. Nothing in the system distinguishes "this fact came from a random
     blog" from "this fact came from Nature / a .gov site / a source
     Tavily itself scored as highly relevant." The final report and the
     Stage-3 grounding check both treat every source's text identically.

This module does NOT call an LLM, does NOT use embeddings, and does NOT
hit any new API -- on purpose, same reasoning as grounding.py: a second
opaque scoring model would just be one more thing to trust blindly. This
is simple, deterministic arithmetic on fields Tavily already returns
(title, url, content, score), so every number it produces can be
recomputed by hand and defended in an interview.

THE FORMULA (this is the whole algorithm):
  For each search result dict {title, url, content, score}:

    relevance   = result["score"] if present and 0<=score<=1, else 0.5
                  (Tavily's own query-relevance score, 0-1. Neutral 0.5
                  default if the field is ever missing, so an unscored
                  result isn't silently punished OR rewarded.)

    domain_trust:
      - no url at all                          -> 0.3
      - domain ends with .gov or .edu           -> 1.0
      - domain is in the curated HIGH_TRUST set -> 1.0
      - domain ends with .org                   -> 0.7
      - anything else (unverified commercial)   -> 0.5

    richness:
      - length_score = min(len(content) / 500, 1.0)
      - has_number   = 1.0 if content contains a digit, else 0.0
      - richness = 0.7 * length_score + 0.3 * has_number
      (Rewards substantive, fact-dense snippets over thin ones. Capped at
      500 chars so one very long result can't dominate the score just by
      being verbose.)

    quality_score = round(0.5*relevance + 0.3*domain_trust + 0.2*richness, 3)
    (clamped to [0, 1])

  Classify: quality_score >= 0.75 -> "High"
            0.5 <= quality_score < 0.75 -> "Medium"
            quality_score < 0.5 -> "Low"

KNOWN, NAMED LIMITATIONS (say this out loud, don't hide it):
  - HIGH_TRUST_DOMAINS is a small, hand-picked allowlist (~20 domains),
    not a general web-authority model. A legitimate high-quality source
    not on the list gets the same 0.5 baseline as a low-quality one --
    this is a deliberate, documented simplification, not an oversight.
  - "richness" measures how much text and how many digits a snippet has,
    not whether the content is actually correct. A long, number-dense
    snippet full of wrong numbers still scores well on richness.
  - Duplicate URLs across the primary and enrichment search are
    deduplicated by URL (first occurrence wins), not by content
    similarity -- two different URLs with near-identical content are
    still counted as two separate sources.
  - The 0.5/0.3/0.2 weights and the 0.6/0.3 domain tiers are reasoned
    defaults (relevance is weighted highest because it is Tavily's own
    measured signal; domain and richness are heuristics on top of it),
    not tuned against a labeled dataset -- there isn't one to tune
    against, same honest caveat as grounding.py's thresholds.
"""

import re
from urllib.parse import urlparse

RELEVANCE_WEIGHT = 0.5
DOMAIN_WEIGHT = 0.3
RICHNESS_WEIGHT = 0.2

HIGH_QUALITY_THRESHOLD = 0.75
MEDIUM_QUALITY_THRESHOLD = 0.5

RICHNESS_LENGTH_CAP = 500  # characters

# Small, hand-picked allowlist of widely-recognized, high-authority
# domains (encyclopedic / scientific / major wire & institutional press).
# NOT exhaustive -- see "Known limitations" above.
HIGH_TRUST_DOMAINS = frozenset({
    "wikipedia.org",
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "nature.com",
    "sciencedirect.com",
    "arxiv.org",
    "who.int",
    "un.org",
    "bbc.com",
    "nytimes.com",
    "economist.com",
    "ft.com",
    "springer.com",
    "ieee.org",
    "nih.gov",
    "cdc.gov",
    "npr.org",
    "science.org",
    "pnas.org",
})


def _domain_of(url: str) -> str:
    """Extracts the bare domain (no scheme, no 'www.', no path) from a
    URL. Returns '' for a missing/unparseable URL rather than raising --
    callers treat '' as "no domain signal available"."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _domain_trust_score(url: str) -> float:
    domain = _domain_of(url)
    if not domain:
        return 0.3
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 1.0
    if domain in HIGH_TRUST_DOMAINS:
        return 1.0
    if domain.endswith(".org"):
        return 0.7
    return 0.5


def _richness_score(content: str) -> float:
    if not content:
        return 0.0
    length_score = min(len(content) / RICHNESS_LENGTH_CAP, 1.0)
    has_number = 1.0 if re.search(r"\d", content) else 0.0
    return 0.7 * length_score + 0.3 * has_number


def _relevance_score(result: dict) -> float:
    score = result.get("score")
    if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
        return float(score)
    return 0.5


def classify_quality(quality_score: float) -> str:
    if quality_score >= HIGH_QUALITY_THRESHOLD:
        return "High"
    elif quality_score >= MEDIUM_QUALITY_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def score_source(result: dict) -> dict:
    """Scores a single raw Tavily result dict (title/url/content/score).

    Returns a new dict -- the original fields plus quality_score (float,
    0-1) and quality_tier (str). Never mutates the input.
    """
    url = result.get("url", "") or ""
    content = result.get("content", "") or ""

    relevance = _relevance_score(result)
    domain_trust = _domain_trust_score(url)
    richness = _richness_score(content)

    quality_score = (
        RELEVANCE_WEIGHT * relevance
        + DOMAIN_WEIGHT * domain_trust
        + RICHNESS_WEIGHT * richness
    )
    quality_score = round(min(max(quality_score, 0.0), 1.0), 3)

    return {
        "title": result.get("title", "") or "(no title)",
        "url": url,
        "content": content,
        "relevance": round(relevance, 3),
        "domain_trust": round(domain_trust, 3),
        "richness": round(richness, 3),
        "quality_score": quality_score,
        "quality_tier": classify_quality(quality_score),
    }


def rank_sources(results: list) -> list:
    """Scores every result, deduplicates by URL (first occurrence wins --
    the primary search result for a URL is kept over a later duplicate
    from an enrichment search), and returns them sorted by quality_score
    descending. Results with no URL are never deduplicated against each
    other (each is kept), since there's no identifier to dedupe on."""
    if not results:
        return []

    seen_urls = set()
    scored = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url", "") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        scored.append(score_source(result))

    scored.sort(key=lambda s: s["quality_score"], reverse=True)
    return scored


def merge_ranked_sources(*ranked_lists) -> list:
    """Merges multiple already-scored/ranked lists (each produced by
    rank_sources()) into one deduplicated, re-sorted list -- WITHOUT
    re-scoring anything. This is deliberately separate from rank_sources(),
    which expects raw Tavily result dicts (title/url/content/score) and
    would silently default every already-scored source's relevance back to
    the neutral 0.5 if fed its own output (since scored dicts no longer
    have a 'score' key, they have 'quality_score'/'relevance' instead).

    Used to combine the primary-search sources and the enrichment-search
    sources (two separate search_web_ranked() calls) into one combined
    view, deduplicated by URL, first-occurrence-wins -- same convention as
    rank_sources()."""
    seen_urls = set()
    merged = []
    for ranked in ranked_lists:
        for s in ranked or []:
            url = s.get("url", "") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            merged.append(s)
    merged.sort(key=lambda s: s["quality_score"], reverse=True)
    return merged


def average_quality(ranked: list) -> float:
    """Mean quality_score across ranked sources, or None if there are no
    sources -- never fabricated as 0, same convention as grounding.py's
    grounding_score."""
    if not ranked:
        return None
    return round(sum(s["quality_score"] for s in ranked) / len(ranked), 3)


def format_ranked_results(ranked: list) -> str:
    """Formats ranked sources as the numbered '[n] Title / content /
    Source: url' text the LLM prompts already expect (same shape as the
    old tools._format_results), but sorted best-first and tagged with a
    quality label so a human skimming research_data/extra_context can see
    at a glance which sources it's most trusting."""
    if not ranked:
        return "No results found."
    formatted = []
    for i, s in enumerate(ranked, 1):
        formatted.append(
            f"[{i}] ({s['quality_tier']} quality, score={s['quality_score']}) {s['title']}\n"
            f"{s['content']}\n"
            f"Source: {s['url']}"
        )
    return "\n\n".join(formatted)


def format_quality_report(ranked: list, label: str = "Sources") -> str:
    """Formats a Markdown summary of source quality for display in the
    Streamlit UI / storage alongside the rest of a session -- same style
    as grounding.format_grounding_report()."""
    if not ranked:
        return f"**{label}:** No sources were retrieved."

    avg = average_quality(ranked)
    high = sum(1 for s in ranked if s["quality_tier"] == "High")
    medium = sum(1 for s in ranked if s["quality_tier"] == "Medium")
    low = sum(1 for s in ranked if s["quality_tier"] == "Low")

    lines = [
        f"**{label}: avg quality {avg}** "
        f"({high} High, {medium} Medium, {low} Low -- out of {len(ranked)} sources)",
        "",
        "_Formula: 0.5 x Tavily relevance + 0.3 x domain trust + 0.2 x content "
        "richness. See source_ranking.py for the full explanation and its "
        "limitations._",
        "",
    ]
    for s in ranked:
        lines.append(
            f"- **[{s['quality_tier']}, {s['quality_score']}]** {s['title']} "
            f"-- {s['url'] or '(no url)'}"
        )
    return "\n".join(lines)
