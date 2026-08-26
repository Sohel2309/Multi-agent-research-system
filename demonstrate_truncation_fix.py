"""
Stage 4: concrete before/after demonstration of the bug ranking fixes.

_extra_searcher_async() (agents.py) truncates research_data to the first
600 characters before handing it to the LLM (research_data[:600]). If a
weak/off-topic source happens to be listed first in Tavily's raw response
order, it can push a strong, on-topic, number-dense source out of that
600-char window entirely -- the LLM never sees it.

This script builds one fixed, realistic set of Tavily-shaped raw results
(one weak, off-topic, thin-content result; one strong, on-topic,
number-dense, .gov result) with the weak one listed FIRST, exactly as
Tavily might return them. It then shows:

  BEFORE (tools._format_results -- Stage <4, unranked, raw Tavily order):
    the strong source's content is truncated out of the 600-char window.

  AFTER (source_ranking.rank_sources + format_ranked_results):
    the strong source is sorted first by quality_score, so it survives
    the same 600-char cutoff.

This does not require network access -- the "raw Tavily results" below
are a fixed, hand-written fixture, not a live API call. Run with:
    python3 demonstrate_truncation_fix.py
"""
from tools import _format_results
from source_ranking import rank_sources, format_ranked_results, score_source

TRUNCATE_LIMIT = 600  # must match agents.py's research_data[:600]

# A weak, off-topic, thin, unknown-domain result Tavily lists FIRST --
# padded with filler text to be long enough to actually eat into the
# 600-char budget on its own, which is realistic: Tavily results are
# often several hundred characters of prose each.
WEAK_SOURCE = {
    "title": "Random blog post loosely related to the topic",
    "url": "https://randomblog.example.com/post-123",
    "content": (
        "This is a general, mostly off-topic discussion with a lot of "
        "filler prose and opinion, no specific figures, no statistics, "
        "and no named sources of its own. It rambles for a while about "
        "tangential context before eventually getting to a vague, "
        "unsupported claim that isn't very useful for research purposes. "
        "There is nothing concrete here for a fact-checker to verify, and "
        "yet the discussion keeps circling back to the same unsupported "
        "generalities without ever citing a single number, date, or named "
        "study, which is exactly the kind of low-value filler that a "
        "research pipeline should not let crowd out a genuinely strong, "
        "number-dense source simply because it happened to be returned "
        "first in an unordered list of raw search results."
    ),
    "score": 0.3,  # Tavily's own relevance score: low
}

# A strong, on-topic, number-dense, .gov (fully trusted) result Tavily
# happens to list SECOND.
STRONG_SOURCE = {
    "title": "Official statistics report",
    "url": "https://www.census.gov/library/report-2025",
    "content": (
        "According to the 2025 report, the measured value increased by "
        "12.4% year over year, reaching 3,821,000 units nationally, up "
        "from 3,400,500 the prior year. Regional breakdowns show a 18% "
        "increase in the Northeast and a 4.2% decline in the Midwest."
    ),
    "score": 0.9,  # Tavily's own relevance score: high
}

RAW_RESULTS = [WEAK_SOURCE, STRONG_SOURCE]  # weak listed first, as Tavily might


def main():
    print("=" * 72)
    print("BEFORE (unranked, raw Tavily order, weak source first)")
    print("=" * 72)
    unranked_formatted = _format_results(RAW_RESULTS)
    unranked_window = unranked_formatted[:TRUNCATE_LIMIT]
    print(unranked_formatted)
    print()
    print(f"--- after research_data[:{TRUNCATE_LIMIT}] truncation, the LLM sees: ---")
    print(unranked_window)
    strong_survives_unranked = "3,821,000" in unranked_window
    print()
    print(f"Strong source's key figure ('3,821,000') present in window? "
          f"{strong_survives_unranked}")

    print()
    print("=" * 72)
    print("AFTER (ranked by quality_score, best source first)")
    print("=" * 72)
    ranked = rank_sources(RAW_RESULTS)
    for r in ranked:
        print(f"  {r['quality_tier']:6s}  quality_score={r['quality_score']:.3f}  "
              f"{r['title']}")
    ranked_formatted = format_ranked_results(ranked)
    ranked_window = ranked_formatted[:TRUNCATE_LIMIT]
    print()
    print(ranked_formatted)
    print()
    print(f"--- after research_data[:{TRUNCATE_LIMIT}] truncation, the LLM sees: ---")
    print(ranked_window)
    strong_survives_ranked = "3,821,000" in ranked_window
    print()
    print(f"Strong source's key figure ('3,821,000') present in window? "
          f"{strong_survives_ranked}")

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    if (not strong_survives_unranked) and strong_survives_ranked:
        print("CONFIRMED: ranking fixes the truncation bug for this fixture.")
        print("  Unranked: strong source's figures cut off by the 600-char window.")
        print("  Ranked:   strong source sorted first, figures survive the same window.")
    else:
        print("NOT CONFIRMED with this fixture -- see raw booleans above. "
              "(This would mean the fixture needs to be adjusted, e.g. the weak "
              "source's content needs to be long enough to actually push the "
              "strong source's figures past character 600.)")

    assert not strong_survives_unranked, (
        "Fixture is not actually demonstrating the bug -- weak source's content "
        "is too short to push the strong source out of the 600-char window."
    )
    assert strong_survives_ranked, (
        "Ranking did not fix it -- check rank_sources()/format_ranked_results()."
    )


if __name__ == "__main__":
    main()
