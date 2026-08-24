"""
Tests for the Stage 2 benchmark harness (benchmark.py, single_agent_baseline.py).

These verify the HARNESS LOGIC is correct -- stats math, JSON structure,
failure handling, and (critically) that LangChain's callback-based token
capture actually reaches the LLM calls buried inside agents.py's node
functions, including the two run concurrently via asyncio.gather.

None of this requires a real GROQ_API_KEY/TAVILY_API_KEY or network access
-- everything here uses mocked search calls and a real LangChain
BaseChatModel subclass (not a plain mock) so the tests exercise the actual
LangChain Runnable/callback machinery, not a stand-in for it.

These tests do NOT tell you whether the multi-agent pipeline is actually
faster/cheaper/better than the single-agent baseline in reality -- only a
real run with real API keys (`python3 benchmark.py`) can tell you that.

Run with: python3 test_benchmark_harness.py
"""
import asyncio
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "dummy_test_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_test_key")

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agents
import graph
import single_agent_baseline
import benchmark


class _RealFakeLLM(BaseChatModel):
    """A genuine LangChain Runnable (not a plain python mock) with known,
    fixed usage_metadata -- used to verify callback propagation and token
    math with numbers we can check by hand."""

    @property
    def _llm_type(self) -> str:
        return "real-fake-for-tests"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = AIMessage(
            content="fake report content",
            usage_metadata={"input_tokens": 111, "output_tokens": 22, "total_tokens": 133},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _fake_search(query, max_results=5):
    return "[1] Title\nContent\nSource: http://example.com"


class TestCallbackTokenCapturePropagatesThroughRealGraph(unittest.TestCase):
    """The single most important correctness check for this stage: does
    config={"callbacks":[...]} passed to graph.ainvoke() actually reach
    every LLM call inside agents.py's unmodified node functions, including
    the ones inside asyncio.gather? If this test breaks, token/cost numbers
    from benchmark.py cannot be trusted and must not be used."""

    def test_all_five_llm_calls_are_captured_with_correct_totals(self):
        cb = benchmark.TokenUsageCallback()
        compiled = graph.build_graph()

        def fake_get_llm():
            return _RealFakeLLM()

        async def run():
            with patch.object(agents, "get_llm", fake_get_llm), \
                 patch.object(agents, "search_web", _fake_search):
                return await compiled.ainvoke(
                    {"query": "test", "error": ""}, config={"callbacks": [cb]}
                )

        result = asyncio.run(run())

        self.assertEqual(cb.llm_call_count, 5, "expected 5 LLM calls: research, analyst, extra_search, writer, qa")
        self.assertEqual(cb.input_tokens, 111 * 5)
        self.assertEqual(cb.output_tokens, 22 * 5)
        self.assertTrue(cb.usage_reliable)
        self.assertTrue(result.get("report"))

    def test_single_agent_baseline_captures_its_one_llm_call(self):
        cb = benchmark.TokenUsageCallback()

        def fake_get_llm():
            return _RealFakeLLM()

        async def run():
            with patch.object(single_agent_baseline, "get_llm", fake_get_llm), \
                 patch.object(single_agent_baseline, "search_web", _fake_search):
                return await single_agent_baseline.run_single_agent("test", config={"callbacks": [cb]})

        result = asyncio.run(run())
        self.assertEqual(cb.llm_call_count, 1)
        self.assertEqual(cb.input_tokens, 111)
        self.assertEqual(result["error"], "")
        self.assertTrue(result["report"])


class TestLatencyStatsMath(unittest.TestCase):
    """Hand-checkable numbers, not trust-me numbers."""

    def test_known_values(self):
        # 10 values, sorted: 1..10. mean=5.5, median=5.5,
        # p95 nearest-rank index = ceil(0.95*10)-1 = 9-1... let's compute: ceil(9.5)=10, idx=10-1=9 -> value 10
        latencies = [float(i) for i in range(1, 11)]
        stats = benchmark.compute_latency_stats(latencies)
        self.assertEqual(stats["n"], 10)
        self.assertEqual(stats["mean"], 5.5)
        self.assertEqual(stats["median_p50"], 5.5)
        self.assertEqual(stats["p95"], 10.0)
        self.assertIsNotNone(stats["note_if_small_n"])  # n=10 < 20

    def test_empty_list_does_not_crash(self):
        stats = benchmark.compute_latency_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])


class TestCostMath(unittest.TestCase):
    def test_known_values(self):
        # 1,000,000 input tokens + 1,000,000 output tokens
        # = $0.15 + $0.60 = $0.75
        cost = benchmark.compute_cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.75, places=6)

    def test_zero_tokens_zero_cost(self):
        self.assertEqual(benchmark.compute_cost(0, 0), 0.0)


class TestBenchmarkFailureHandling(unittest.TestCase):
    """A failed run must be marked unsuccessful, must not fabricate a
    report, and must NOT count toward latency stats (its wall-clock time
    is mostly Stage-1 retry backoff, not report-generation time)."""

    def test_search_failure_excluded_from_latency_and_not_fabricated(self):
        def failing_search(query, max_results=5):
            raise agents.SearchError("simulated failure")

        def fake_get_llm():
            return _RealFakeLLM()

        async def run():
            with patch.object(agents, "get_llm", fake_get_llm), \
                 patch.object(agents, "search_web", failing_search), \
                 patch.object(single_agent_baseline, "get_llm", fake_get_llm), \
                 patch.object(single_agent_baseline, "search_web", failing_search):
                return await benchmark._execute_all(["q1"], trials=1, pipeline="both")

        results = asyncio.run(run())
        summary = benchmark.summarize(results, dry_run=True)

        multi = [r for r in results if r["pipeline"] == "multi_agent"][0]
        self.assertFalse(multi["success"])
        self.assertEqual(multi["report_char_length"], 0)
        self.assertNotEqual(multi["error"], "")
        self.assertEqual(summary["multi_agent"]["n_success"], 0)
        self.assertEqual(summary["multi_agent"]["latency_seconds"]["n"], 0)
        self.assertIsNone(summary["multi_agent"]["total_llm_cost_usd"])


class TestDryRunEndToEnd(unittest.TestCase):
    """Exercises the full CLI-driven path (run_benchmark -> summarize ->
    save_results) exactly as `python3 benchmark.py --dry-run` would,
    including file I/O, then cleans up after itself."""

    def setUp(self):
        self.test_results_dir = Path(__file__).parent / "benchmark_results"
        self._existing_before = set(self.test_results_dir.glob("*.json")) if self.test_results_dir.exists() else set()

    def tearDown(self):
        if self.test_results_dir.exists():
            new_files = set(self.test_results_dir.glob("*.json")) - self._existing_before
            for f in new_files:
                f.unlink()
            if not any(self.test_results_dir.iterdir()):
                self.test_results_dir.rmdir()

    def test_full_dry_run_produces_valid_output_files(self):
        queries = benchmark.QUERIES[:2]
        raw = asyncio.run(benchmark.run_benchmark(queries, trials=1, pipeline="both", dry_run=True))
        self.assertEqual(len(raw), 4)  # 2 queries x 2 pipelines x 1 trial
        summary = benchmark.summarize(raw, dry_run=True)
        self.assertIn("multi_agent", summary)
        self.assertIn("single_agent", summary)
        self.assertTrue(summary["dry_run"])

        benchmark.save_results(raw, summary, dry_run=True)
        raw_files = list(self.test_results_dir.glob("dryrun_raw_*.json"))
        summary_files = list(self.test_results_dir.glob("dryrun_summary_*.json"))
        self.assertEqual(len(raw_files), 1)
        self.assertEqual(len(summary_files), 1)

        with open(raw_files[0]) as f:
            saved_raw = json.load(f)
        self.assertEqual(len(saved_raw["runs"]), 4)
        self.assertTrue(saved_raw["dry_run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
