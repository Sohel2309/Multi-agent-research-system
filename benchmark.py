"""
Stage 2: Reproducible benchmark/evaluation harness.

Runs the fixed query set (benchmark_queries.QUERIES) through both the
multi-agent pipeline (graph.py, unmodified) and the single-agent baseline
(single_agent_baseline.py) and records, per run:
  - wall-clock latency (time.perf_counter -- always trustworthy)
  - success/failure (from the pipeline's own error field / exceptions)
  - LLM token usage, IF available (via a LangChain callback attached to
    graph.ainvoke()/llm.ainvoke() -- see TokenUsageCallback below)
  - cost, ONLY IF token usage was actually captured for that run

Nothing here modifies agents.py, graph.py, tools.py, state.py, database.py,
or app.py. This script only imports and calls their existing, public
functions (build_graph, get_llm, search_web) exactly as they already work.

USAGE (needs real GROQ_API_KEY / TAVILY_API_KEY in your environment):
    python3 benchmark.py                    # all 8 queries, both pipelines, 1 trial each
    python3 benchmark.py --queries 3         # first 3 queries only (cheap smoke test)
    python3 benchmark.py --trials 3          # 3 trials per query per pipeline (for variance)
    python3 benchmark.py --pipeline multi    # only the multi-agent pipeline
    python3 benchmark.py --pipeline single   # only the single-agent baseline

USAGE (no API keys/network needed -- validates the harness itself):
    python3 benchmark.py --dry-run

Every run writes two files under benchmark_results/:
    raw_<timestamp>.json       one row per individual pipeline run (reproducibility)
    summary_<timestamp>.json   aggregated stats used for the comparison table

IMPORTANT: This sandboxed development environment has no network access to
api.groq.com/api.tavily.com, so every number in the accompanying report was
either produced with --dry-run (mocked, clearly labeled, proves the harness
logic is correct) or is explicitly marked "not run." Real benchmark numbers
require running this script with real API keys -- see test_benchmark_harness.py
for what has already been verified without them.
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agents
import single_agent_baseline
from graph import build_graph
from benchmark_queries import QUERIES

RESULTS_DIR = Path(__file__).parent / "benchmark_results"

# Pricing for openai/gpt-oss-120b on Groq, confirmed from
# https://console.groq.com/docs/models.md on 2026-08-24 (Production Models
# table): $0.15 / 1M input tokens, $0.60 / 1M output tokens. Groq can change
# pricing at any time -- re-verify this before trusting cost numbers from an
# old benchmark run for anything beyond historical reference.
GPT_OSS_120B_INPUT_COST_PER_1M = 0.15
GPT_OSS_120B_OUTPUT_COST_PER_1M = 0.60
PRICING_SOURCE = "https://console.groq.com/docs/models.md (checked 2026-08-24)"

# This only prices LLM tokens. Tavily search-call cost is NOT included --
# no reliably-sourced per-request Tavily price was available when this was
# written (tools.py's own comment only notes "free tier: 1000 searches/month").
# Don't add a number here without a citation you can point to in an interview.


# ─────────────────────────────────────────────────────────────────────────
# Token usage capture
# ─────────────────────────────────────────────────────────────────────────

class TokenUsageCallback(AsyncCallbackHandler):
    """Captures token usage from every LLM call inside a graph run, without
    touching agents.py/graph.py at all -- attached via
    config={"callbacks": [...]} on graph.ainvoke() / llm.ainvoke().

    This relies on two things, both verified in test_benchmark_harness.py:
      1. LangChain callbacks passed at the top-level ainvoke() call DO
         propagate down into node functions' internal llm.ainvoke() calls,
         including the two that run concurrently inside asyncio.gather in
         agents.parallel_step().
      2. ChatGroq populates AIMessage.usage_metadata from Groq's real API
         response (confirmed by reading langchain_groq's installed source,
         not assumed).
    If usage_metadata is ever missing on a real run, this is reported
    honestly (token_usage_available=False) rather than silently showing 0.
    """

    def __init__(self):
        self.llm_call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.saw_missing_usage = False

    async def on_llm_end(self, response, **kwargs):
        self.llm_call_count += 1
        found_usage = False
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    found_usage = True
                    self.input_tokens += usage.get("input_tokens", 0) or 0
                    self.output_tokens += usage.get("output_tokens", 0) or 0
                    self.total_tokens += usage.get("total_tokens", 0) or 0
        if not found_usage:
            self.saw_missing_usage = True

    @property
    def usage_reliable(self) -> bool:
        return self.llm_call_count > 0 and not self.saw_missing_usage


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * GPT_OSS_120B_INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * GPT_OSS_120B_OUTPUT_COST_PER_1M
    )


# ─────────────────────────────────────────────────────────────────────────
# Latency stats -- pure stdlib, no numpy/pandas (keeping this simple and
# fully explainable: nearest-rank percentile, the same method taught in
# most intro stats/SRE material)
# ─────────────────────────────────────────────────────────────────────────

def compute_latency_stats(latencies: list) -> dict:
    if not latencies:
        return {"n": 0, "mean": None, "median_p50": None, "p95": None}
    ordered = sorted(latencies)
    n = len(ordered)
    idx_p95 = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {
        "n": n,
        "mean": round(statistics.mean(ordered), 3),
        "median_p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[idx_p95], 3),
        "note_if_small_n": (
            "n < 20 -- p95 here is not statistically robust, treat as directional only"
            if n < 20 else None
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Running a single query through each pipeline
# ─────────────────────────────────────────────────────────────────────────

async def _run_multi_agent_once(query: str) -> dict:
    initial_state = {
        "query": query, "research_data": "", "analysis": "", "extra_context": "",
        "report": "", "qa_review": "", "messages": [], "error": "",
    }
    cb = TokenUsageCallback()
    compiled = build_graph()
    start = time.perf_counter()
    try:
        result = await compiled.ainvoke(initial_state, config={"callbacks": [cb]})
        elapsed = time.perf_counter() - start
        success = not bool(result.get("error"))
        return {
            "pipeline": "multi_agent",
            "query": query,
            "success": success,
            "error": result.get("error", ""),
            "latency_seconds": round(elapsed, 3),
            "llm_call_count": cb.llm_call_count,
            "input_tokens": cb.input_tokens,
            "output_tokens": cb.output_tokens,
            "total_tokens": cb.total_tokens,
            "token_usage_available": cb.usage_reliable,
            "report_char_length": len(result.get("report", "")),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "pipeline": "multi_agent", "query": query, "success": False,
            "error": f"unhandled exception: {e}", "latency_seconds": round(elapsed, 3),
            "llm_call_count": cb.llm_call_count, "input_tokens": cb.input_tokens,
            "output_tokens": cb.output_tokens, "total_tokens": cb.total_tokens,
            "token_usage_available": False, "report_char_length": 0,
        }


async def _run_single_agent_once(query: str) -> dict:
    cb = TokenUsageCallback()
    start = time.perf_counter()
    try:
        result = await single_agent_baseline.run_single_agent(query, config={"callbacks": [cb]})
        elapsed = time.perf_counter() - start
        success = not bool(result.get("error"))
        return {
            "pipeline": "single_agent",
            "query": query,
            "success": success,
            "error": result.get("error", ""),
            "latency_seconds": round(elapsed, 3),
            "llm_call_count": cb.llm_call_count,
            "input_tokens": cb.input_tokens,
            "output_tokens": cb.output_tokens,
            "total_tokens": cb.total_tokens,
            "token_usage_available": cb.usage_reliable,
            "report_char_length": len(result.get("report", "")),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "pipeline": "single_agent", "query": query, "success": False,
            "error": f"unhandled exception: {e}", "latency_seconds": round(elapsed, 3),
            "llm_call_count": cb.llm_call_count, "input_tokens": cb.input_tokens,
            "output_tokens": cb.output_tokens, "total_tokens": cb.total_tokens,
            "token_usage_available": False, "report_char_length": 0,
        }


# ─────────────────────────────────────────────────────────────────────────
# Dry-run mocks (no network / API keys needed) -- used to validate the
# harness itself. A REAL LangChain BaseChatModel subclass is used (not a
# plain function), so the same callback machinery that would run against
# the real ChatGroq is genuinely exercised here too.
# ─────────────────────────────────────────────────────────────────────────

class _DryRunFakeLLM(BaseChatModel):
    """Deterministic-ish fake LLM for --dry-run. Adds a small random-ish
    (but seeded, reproducible) delay so latency stats have something real
    to compute, and reports fake-but-realistic usage_metadata so the token
    capture path is exercised end to end, not skipped."""

    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "dry-run-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("sync path unused")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # deterministic pseudo-variance so p50 != p95 in the dry-run demo
        self.call_index += 1
        await asyncio.sleep(0.01 + (self.call_index % 5) * 0.005)
        msg = AIMessage(
            content=f"[DRY RUN fake report content, call #{self.call_index}]",
            usage_metadata={"input_tokens": 400, "output_tokens": 150, "total_tokens": 550},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _dry_run_search_web(query: str, max_results: int = 5) -> str:
    return f"[DRY RUN fake search result for: {query}]\n[1] Fake Title\nFake content\nSource: http://example.com"


def _make_dry_run_get_llm():
    def _get_llm():
        return _DryRunFakeLLM()
    return _get_llm


# ─────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────

async def _execute_all(queries: list, trials: int, pipeline: str) -> list:
    results = []
    for q in queries:
        for t in range(1, trials + 1):
            if pipeline in ("multi", "both"):
                r = await _run_multi_agent_once(q)
                r["trial"] = t
                results.append(r)
                print(f"[multi_agent]  trial {t}/{trials}  '{q[:55]}'  "
                      f"success={r['success']}  {r['latency_seconds']}s")
            if pipeline in ("single", "both"):
                r = await _run_single_agent_once(q)
                r["trial"] = t
                results.append(r)
                print(f"[single_agent] trial {t}/{trials}  '{q[:55]}'  "
                      f"success={r['success']}  {r['latency_seconds']}s")
    return results


async def run_benchmark(queries: list, trials: int, pipeline: str, dry_run: bool) -> list:
    if dry_run:
        fake_get_llm = _make_dry_run_get_llm()
        with patch.object(agents, "get_llm", fake_get_llm), \
             patch.object(agents, "search_web", _dry_run_search_web), \
             patch.object(single_agent_baseline, "get_llm", fake_get_llm), \
             patch.object(single_agent_baseline, "search_web", _dry_run_search_web):
            return await _execute_all(queries, trials, pipeline)
    else:
        return await _execute_all(queries, trials, pipeline)


def summarize(raw_results: list, dry_run: bool) -> dict:
    summary = {"dry_run": dry_run, "pricing_source": PRICING_SOURCE}
    for pipeline_name in sorted(set(r["pipeline"] for r in raw_results)):
        rows = [r for r in raw_results if r["pipeline"] == pipeline_name]
        successes = [r for r in rows if r["success"]]
        # Only successful runs count toward latency stats: a failed run's
        # wall-clock time mostly measures Stage-1 retry backoff (2s-10s per
        # attempt), not "how long does it take to get a report" -- mixing
        # them in would misrepresent both numbers.
        latencies = [r["latency_seconds"] for r in successes]
        token_rows = [r for r in successes if r["token_usage_available"]]
        total_input = sum(r["input_tokens"] for r in token_rows)
        total_output = sum(r["output_tokens"] for r in token_rows)
        cost = compute_cost(total_input, total_output) if token_rows else None

        summary[pipeline_name] = {
            "n_runs": len(rows),
            "n_success": len(successes),
            "success_rate": round(len(successes) / len(rows), 3) if rows else None,
            "latency_seconds": compute_latency_stats(latencies),
            "total_llm_calls": sum(r["llm_call_count"] for r in rows),
            "avg_llm_calls_per_run": (
                round(sum(r["llm_call_count"] for r in rows) / len(rows), 2) if rows else None
            ),
            "token_usage_coverage": f"{len(token_rows)}/{len(successes)} successful runs had usable token data",
            "total_input_tokens": total_input if token_rows else None,
            "total_output_tokens": total_output if token_rows else None,
            "total_llm_cost_usd": round(cost, 6) if cost is not None else None,
            "avg_llm_cost_per_successful_run_usd": (
                round(cost / len(token_rows), 6) if cost is not None and token_rows else None
            ),
            "avg_report_char_length": (
                round(statistics.mean([r["report_char_length"] for r in successes]), 1)
                if successes else None
            ),
        }
    return summary


def print_comparison_table(summary: dict):
    if "multi_agent" not in summary or "single_agent" not in summary:
        return  # only one pipeline was run this time -- nothing to compare
    m = summary["multi_agent"]
    s = summary["single_agent"]
    print("\n" + "=" * 78)
    print("MULTI-AGENT vs SINGLE-AGENT")
    print("=" * 78)
    rows = [
        ("Success rate", m["success_rate"], s["success_rate"]),
        ("Mean latency (s)", m["latency_seconds"]["mean"], s["latency_seconds"]["mean"]),
        ("p50 latency (s)", m["latency_seconds"]["median_p50"], s["latency_seconds"]["median_p50"]),
        ("p95 latency (s)", m["latency_seconds"]["p95"], s["latency_seconds"]["p95"]),
        ("Avg LLM calls/run", m["avg_llm_calls_per_run"], s["avg_llm_calls_per_run"]),
        ("Total LLM cost (USD)", m["total_llm_cost_usd"], s["total_llm_cost_usd"]),
        ("Avg report length (chars)", m["avg_report_char_length"], s["avg_report_char_length"]),
    ]
    print(f"{'Metric':<28}{'Multi-Agent':>18}{'Single-Agent':>18}")
    for label, mv, sv in rows:
        print(f"{label:<28}{str(mv):>18}{str(sv):>18}")
    print("=" * 78)
    print("NOTE: This table does NOT include any report-quality/accuracy metric.")
    print("Quality/groundedness comparison is Stage 3 (grounded QA), not this stage.")
    print("=" * 78 + "\n")


def save_results(raw_results: list, summary: dict, dry_run: bool):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "dryrun_" if dry_run else ""
    raw_path = RESULTS_DIR / f"{prefix}raw_{ts}.json"
    summary_path = RESULTS_DIR / f"{prefix}summary_{ts}.json"

    with open(raw_path, "w") as f:
        json.dump({"generated_at_utc": ts, "dry_run": dry_run, "runs": raw_results}, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump({"generated_at_utc": ts, **summary}, f, indent=2)

    print(f"Raw per-run results saved to:     {raw_path}")
    print(f"Aggregated summary saved to:      {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 benchmark harness: multi-agent vs single-agent")
    p.add_argument("--queries", type=int, default=None,
                    help="Use only the first N queries from benchmark_queries.QUERIES (default: all %d)" % len(QUERIES))
    p.add_argument("--trials", type=int, default=1,
                    help="Repeated trials per query per pipeline (default: 1). Increase for more meaningful p95.")
    p.add_argument("--pipeline", choices=["multi", "single", "both"], default="both")
    p.add_argument("--dry-run", action="store_true",
                    help="Use mocked LLM/search calls -- no API keys or network needed. "
                         "Validates the harness logic; does NOT produce real benchmark numbers.")
    return p.parse_args()


def main():
    args = parse_args()
    queries = QUERIES[: args.queries] if args.queries else QUERIES

    if args.dry_run:
        print("=" * 78)
        print("DRY RUN MODE -- using mocked LLM/search calls, NOT real Groq/Tavily.")
        print("These numbers validate the harness logic only. They are NOT real")
        print("benchmark results and must never be quoted as resume metrics.")
        print("=" * 78)
    else:
        import os
        missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
        if missing:
            print(f"ERROR: missing required environment variable(s): {', '.join(missing)}")
            print("Set them, or run with --dry-run to validate the harness without real keys.")
            sys.exit(1)

    print(f"Queries: {len(queries)} | Trials/query/pipeline: {args.trials} | Pipeline(s): {args.pipeline}\n")

    raw_results = asyncio.run(run_benchmark(queries, args.trials, args.pipeline, args.dry_run))
    summary = summarize(raw_results, args.dry_run)

    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))
    print_comparison_table(summary)
    save_results(raw_results, summary, args.dry_run)


if __name__ == "__main__":
    main()
