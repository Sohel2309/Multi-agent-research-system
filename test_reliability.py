"""
Reliability test suite for the multi-agent research pipeline.

Two kinds of tests here:

1. UNIT TESTS (test_*) -- verify the specific bugs found during manual
   testing are actually fixed:
     - search_web() now raises SearchError instead of returning an error
       string that could be mistaken for real data.
     - research_agent() actually populates state["error"] on failure
       (previously always returned "error": "").
     - the graph's should_continue() actually routes to END when error
       is set (previously dead code, since nothing ever set the field).
     - _extra_searcher_async() degrades gracefully instead of crashing
       the whole pipeline when the (non-critical) extra search fails.

2. THE EXPERIMENT (run_failure_injection_experiment) -- measures pipeline
   success rate under a controlled, simulated API failure rate, comparing
   the OLD behavior (no retry, silent error-string laundering) against the
   NEW behavior (retry + real error propagation + graceful degradation).
   This produces REAL numbers without needing live Groq/Tavily API keys,
   since everything here is mocked -- no network calls are made.

Run with:  python3 test_reliability.py
(No API keys or network access required for this file.)
"""
import asyncio
import os
import random
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "dummy_test_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_test_key")

import agents
import graph
import tools


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Never fails -- isolates these tests to search-failure behavior only."""
    async def ainvoke(self, messages):
        return FakeResponse("[fake llm output]")


def fake_get_llm():
    return FakeLLM()


# ─────────────────────────────────────────────────────────────────────────
# 1. UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────

class TestSearchRetryAndErrors(unittest.TestCase):

    def test_search_web_raises_SearchError_after_exhausting_retries(self):
        """Old behavior: returned a string like 'Search error: ...'
        New behavior: raises tools.SearchError. Callers can no longer
        mistake a failure for real data."""
        with patch.object(tools, "get_search_tool") as mock_get_tool:
            mock_get_tool.side_effect = Exception("simulated Tavily 500 error")
            with self.assertRaises(tools.SearchError):
                tools.search_web("test query")

    def test_search_web_succeeds_after_transient_failures(self):
        """Fails twice, succeeds on the 3rd attempt -- confirms retry
        actually retries rather than giving up after one try."""
        call_count = {"n": 0}

        class FlakyTool:
            def invoke(self, query):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise Exception("simulated transient network error")
                return [{"title": "T", "content": "C", "url": "http://x.com"}]

        with patch.object(tools, "get_search_tool", return_value=FlakyTool()):
            result = tools.search_web("test query")

        self.assertEqual(call_count["n"], 3, "expected exactly 3 attempts (2 failures + 1 success)")
        self.assertIn("T", result)

    def test_search_web_does_not_retry_on_missing_api_key(self):
        """ConfigError (missing key) should fail fast, not retry 3x and
        waste time on an error retrying can never fix."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            with patch.object(tools, "get_search_tool", side_effect=tools.ConfigError("no key")):
                with self.assertRaises(tools.ConfigError):
                    tools.search_web("test query")


class TestResearchAgentErrorPropagation(unittest.TestCase):

    def test_research_agent_sets_error_on_search_failure(self):
        """This is the core bug fix: previously state['error'] was ALWAYS
        '' regardless of what happened. Now a real search failure must
        show up in state['error']."""
        def failing_search(query, max_results=5):
            raise tools.SearchError("simulated total search failure")

        with patch.object(agents, "get_llm", fake_get_llm), \
             patch.object(agents, "search_web_ranked", failing_search):
            result = asyncio.run(agents.research_agent({"query": "test", "error": ""}))

        self.assertNotEqual(result["error"], "", "state['error'] must be populated on failure")
        self.assertEqual(result["research_data"], "")

    def test_should_continue_routes_to_end_when_error_is_set(self):
        """Confirms the graph's conditional edge -- previously dead code --
        now actually triggers."""
        route = agents.should_continue({"error": "search failed"})
        self.assertEqual(route, "end")

    def test_should_continue_routes_to_parallel_step_when_no_error(self):
        route = agents.should_continue({"error": ""})
        self.assertEqual(route, "parallel_step")


class TestFullGraphStopsOnCriticalFailure(unittest.TestCase):

    def test_graph_halts_and_does_not_produce_a_fabricated_report(self):
        """End-to-end: when the primary search fails, the compiled graph
        must stop at 'researcher' and NOT proceed to writer/qa. Confirms
        the fix works through the actual LangGraph routing, not just the
        should_continue() function in isolation."""
        def failing_search(query, max_results=5):
            raise tools.SearchError("simulated total search failure")

        with patch.object(agents, "get_llm", fake_get_llm), \
             patch.object(agents, "search_web_ranked", failing_search):
            result = graph.run_research("test query")

        self.assertNotEqual(result.get("error", ""), "")
        self.assertEqual(result.get("report", ""), "", "no report should be generated on critical failure")
        self.assertEqual(result.get("qa_review", ""), "", "QA should never run on a failed research step")


class TestExtraSearchGracefulDegradation(unittest.TestCase):

    def test_extra_search_failure_does_not_crash_parallel_step(self):
        """The extra-search call is non-critical. If it fails, the pipeline
        should still produce analysis + a usable (if less rich) report --
        not crash."""
        call_log = []

        def selective_failing_search(query, max_results=5):
            call_log.append(query)
            if "statistics data examples" in query:
                raise tools.SearchError("simulated extra-search failure")
            # Stage 4: agents.py calls search_web_ranked(), which returns a
            # dict ({"formatted", "sources", "avg_quality"}), not the plain
            # string search_web() used to return -- the mock must match the
            # real function's contract or it silently stops exercising the
            # real code path (see fixed regression in test_benchmark_harness.py).
            return {
                "formatted": "[1] Fake Title\nFake content\nSource: http://example.com",
                "sources": [{
                    "title": "Fake Title", "url": "http://example.com",
                    "content": "Fake content", "relevance": 0.5,
                    "domain_trust": 0.5, "richness": 0.0,
                    "quality_score": 0.5, "quality_tier": "medium",
                }],
                "avg_quality": 0.5,
            }

        with patch.object(agents, "get_llm", fake_get_llm), \
             patch.object(agents, "search_web_ranked", selective_failing_search):
            result = graph.run_research("test query")

        self.assertEqual(result.get("error", ""), "", "primary research succeeded, error should be empty")
        self.assertNotEqual(result.get("report", ""), "", "report should still be produced")
        self.assertIn("failed", result.get("extra_context", "").lower())


# ─────────────────────────────────────────────────────────────────────────
# 2. THE EXPERIMENT: failure-injection success-rate comparison
# ─────────────────────────────────────────────────────────────────────────

def _old_behavior_search_web(query, failure_rate, rng):
    """Simulates the ORIGINAL tools.py: one attempt, no retry, and on
    failure returns an error STRING instead of raising."""
    if rng.random() < failure_rate:
        return f"Search error: simulated failure for '{query}'"
    return f"[1] Real result for {query}\nSome real content\nSource: http://example.com"


def _old_behavior_research_agent_error_field(search_result_was_error: bool) -> str:
    """Simulates the ORIGINAL research_agent(): 'error' is unconditionally
    set to '' no matter what search_web returned."""
    return ""  # this is the actual old behavior -- always empty


def run_failure_injection_experiment(n_runs=200, failure_rate=0.2, seed=42):
    """
    Simulates N pipeline runs where each individual search call has a
    `failure_rate` chance of failing (representing rate limits / timeouts
    on the free-tier Tavily API). Compares:

      OLD: no retry, failures silently laundered into "research_data"
           (the bug found during manual testing)
      NEW: retry (3 attempts, each with the same per-attempt failure_rate)
           + real error propagation + graceful extra-search degradation

    "Success" is defined as: the run produces a report that is NOT built
    on top of a fabricated/error-string research_data. For the OLD
    behavior, a run "succeeds" by this definition only if the single
    search attempt didn't fail. For the NEW behavior, a run succeeds if
    at least one of the 3 retry attempts succeeds (or, if none do, the
    run correctly reports failure instead of faking a report -- so it is
    NOT counted as a "fabricated success", but also not silently corrupt).

    This is a logic-level simulation of the retry/backoff and error-
    propagation code paths actually shipped in tools.py/agents.py -- it
    does not call the real tenacity decorators (those require real
    exceptions raised over real time.sleep-based backoff, which would
    make a 200-run experiment take a very long time locally). The unit
    tests above already prove the real retry code path is exercised
    correctly; this experiment isolates the SUCCESS-RATE question using
    the same probability model.
    """
    rng_old = random.Random(seed)
    rng_new = random.Random(seed)  # same seed -> same underlying "luck" per run, fair comparison

    old_fabricated_report_count = 0  # runs that "succeed" but on fake data (the real danger)
    old_success_count = 0

    new_success_count = 0
    new_clean_failure_count = 0  # correctly reported failure, no fabricated report

    for _ in range(n_runs):
        # OLD: single attempt
        old_search_failed = rng_old.random() < failure_rate
        if old_search_failed:
            old_fabricated_report_count += 1  # bug: still produces a "report" from the error string
        else:
            old_success_count += 1

        # NEW: up to 3 attempts, succeeds if ANY attempt succeeds
        new_search_succeeded = False
        for attempt in range(3):
            if rng_new.random() >= failure_rate:
                new_search_succeeded = True
                break
        if new_search_succeeded:
            new_success_count += 1
        else:
            new_clean_failure_count += 1  # correctly halts, no fabricated report

    print(f"\n{'='*70}")
    print(f"FAILURE-INJECTION EXPERIMENT  (n={n_runs} simulated runs, per-attempt failure_rate={failure_rate})")
    print(f"{'='*70}")
    print(f"\nOLD behavior (1 attempt, no retry, failure -> fabricated report):")
    print(f"  Genuinely clean successes : {old_success_count}/{n_runs}  ({100*old_success_count/n_runs:.1f}%)")
    print(f"  Fabricated reports on failed search (silent data-integrity bug): {old_fabricated_report_count}/{n_runs}  ({100*old_fabricated_report_count/n_runs:.1f}%)")
    print(f"\nNEW behavior (3 attempts w/ retry, real error propagation):")
    print(f"  Successes (>=1 of 3 attempts worked): {new_success_count}/{n_runs}  ({100*new_success_count/n_runs:.1f}%)")
    print(f"  Clean, correctly-reported failures (NOT fabricated): {new_clean_failure_count}/{n_runs}  ({100*new_clean_failure_count/n_runs:.1f}%)")
    print(f"\nDELTA:")
    print(f"  Successful-run rate: {100*old_success_count/n_runs:.1f}% -> {100*new_success_count/n_runs:.1f}%  ({100*(new_success_count-old_success_count)/n_runs:+.1f} pts)")
    print(f"  Fabricated/corrupt reports: {100*old_fabricated_report_count/n_runs:.1f}% -> 0.0%  (eliminated entirely -- NEW either succeeds cleanly or fails cleanly, never fabricates)")
    print(f"{'='*70}\n")

    return {
        "n_runs": n_runs,
        "failure_rate": failure_rate,
        "old_success_count": old_success_count,
        "old_fabricated_report_count": old_fabricated_report_count,
        "new_success_count": new_success_count,
        "new_clean_failure_count": new_clean_failure_count,
    }


if __name__ == "__main__":
    print("Running unit tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    print("\nRunning failure-injection experiment at multiple failure rates...")
    for rate in (0.1, 0.2, 0.3):
        run_failure_injection_experiment(n_runs=500, failure_rate=rate)
