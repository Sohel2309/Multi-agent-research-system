"""
Fixed set of representative research queries for benchmarking the
multi-agent pipeline vs. the single-agent baseline.

Design notes (so this is defensible in an interview):
- Kept small (8 queries) deliberately -- Groq's free/developer tier is
  rate-limited (noted in this project's own README), and each query fires
  multiple LLM calls per pipeline. A larger fixed set is easy to add later
  (just append to QUERIES) once rate-limit headroom is confirmed.
- Spans 4 categories on purpose, so the benchmark isn't accidentally
  measuring "how well does this system handle one kind of topic":
    - emerging/technical  (quantum computing, CRISPR, autonomous vehicles)
    - social/economic     (remote work, AI and jobs)
    - comparative         (solar vs wind)
    - contested/ambiguous (social media and mental health -- useful later
      for Stage 3 grounded-QA work, kept here now for continuity)
- Two of these (quantum computing, EV adoption in India) intentionally
  overlap with the queries already used in this repo's
  measure_time.py / measure_hallucination.py scripts, so results are
  loosely comparable to the project's pre-existing (pre-migration)
  informal numbers, not just internally consistent with themselves.

This list is intentionally static (not randomly sampled) so that re-running
the benchmark on a different day/model is comparing the same inputs --
that's what "reproducible" means here.
"""

QUERIES = [
    "Latest developments in quantum computing",
    "Impact of remote work on employee productivity",
    "Growth of electric vehicle adoption in India",
    "How does CRISPR gene editing work",
    "Comparison of solar vs wind renewable energy",
    "Effects of social media on teenage mental health",
    "Current state of autonomous vehicle technology",
    "Economic impact of artificial intelligence on jobs",
]
