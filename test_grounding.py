"""
Tests for Stage 3: grounded fact-checking (grounding.py + its integration
into agents.qa_agent, state.py, database.py, and the full graph).

Covers, per the Stage 3 requirements:
  - Supported / Partially Supported / Unsupported claim classification
  - Claim extraction edge cases (empty report, header-only report,
    decimal numbers, bullet lists)
  - Failure cases (empty sources, no checkable claims -- must return
    None, never a fabricated 0)
  - Integration into qa_agent (mocked LLM) without breaking qa_review
  - Integration into the full compiled graph (mocked), confirming
    grounding fields survive LangGraph's state schema
  - Database migration: an old pre-Stage-3 sessions.db gains the new
    columns without losing existing data
  - A mocked "full pipeline run" containing one real, source-backed claim
    and one fabricated/hallucinated claim, to demonstrate the feature
    behaves sensibly end-to-end, not just unit-by-unit

None of this requires a real GROQ_API_KEY/TAVILY_API_KEY or network
access -- grounding.py itself never calls any LLM or search API, and all
LLM/search calls in the integration tests are mocked.

Run with: python3 test_grounding.py
"""
import asyncio
import os
import sqlite3
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "dummy_test_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_test_key")

import grounding
import agents
import graph
import database


# ─────────────────────────────────────────────────────────────────────────
# Claim extraction
# ─────────────────────────────────────────────────────────────────────────

class TestClaimExtraction(unittest.TestCase):

    def test_empty_report_returns_no_claims(self):
        self.assertEqual(grounding.extract_claims(""), [])
        self.assertEqual(grounding.extract_claims("   "), [])

    def test_headers_are_not_treated_as_claims(self):
        report = "# Quantum Computing Report\n## Executive Summary\n### Key Findings"
        self.assertEqual(grounding.extract_claims(report), [])

    def test_short_fragments_are_dropped(self):
        report = "- OK\n- Yes\n- Good\n- Sure thing"
        # all fewer than MIN_CLAIM_WORDS (5) words
        self.assertEqual(grounding.extract_claims(report), [])

    def test_bullet_points_are_extracted_as_claims(self):
        report = "- IBM plans to release a 1000-qubit processor by 2025."
        claims = grounding.extract_claims(report)
        self.assertEqual(len(claims), 1)
        self.assertIn("IBM plans to release", claims[0])
        self.assertNotIn("-", claims[0][:2])  # leading bullet marker stripped

    def test_decimal_numbers_are_not_split_mid_sentence(self):
        report = "The market grew by 3.5% in the last quarter according to the report."
        claims = grounding.extract_claims(report)
        self.assertEqual(len(claims), 1)
        self.assertIn("3.5%", claims[0])

    def test_two_sentences_on_one_line_are_split(self):
        report = "Google released Willow in 2024. IBM released a 1000-qubit chip in 2025."
        claims = grounding.extract_claims(report)
        self.assertEqual(len(claims), 2)

    def test_bold_and_italic_markers_are_stripped_but_text_kept(self):
        report = "**IBM** announced a *major* new quantum processor this year with big implications."
        claims = grounding.extract_claims(report)
        self.assertEqual(len(claims), 1)
        self.assertNotIn("**", claims[0])
        self.assertNotIn("*", claims[0])
        self.assertIn("IBM", claims[0])


# ─────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────

class TestTokenize(unittest.TestCase):

    def test_stopwords_removed(self):
        tokens = grounding._tokenize("This is a report about the quantum computing industry")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("quantum", tokens)
        self.assertIn("computing", tokens)
        self.assertIn("industry", tokens)

    def test_case_insensitive(self):
        self.assertEqual(grounding._tokenize("Quantum COMPUTING"), grounding._tokenize("quantum computing"))

    def test_numbers_and_percent_kept(self):
        tokens = grounding._tokenize("Growth reached 51% in 2025")
        self.assertIn("51%", tokens)
        self.assertIn("2025", tokens)

    def test_empty_string(self):
        self.assertEqual(grounding._tokenize(""), set())


# ─────────────────────────────────────────────────────────────────────────
# Classification thresholds
# ─────────────────────────────────────────────────────────────────────────

class TestClassifyOverlap(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(grounding.classify_overlap(1.0), "Supported")
        self.assertEqual(grounding.classify_overlap(0.6), "Supported")  # boundary, inclusive
        self.assertEqual(grounding.classify_overlap(0.59), "Partially Supported")
        self.assertEqual(grounding.classify_overlap(0.3), "Partially Supported")  # boundary, inclusive
        self.assertEqual(grounding.classify_overlap(0.29), "Unsupported")
        self.assertEqual(grounding.classify_overlap(0.0), "Unsupported")


# ─────────────────────────────────────────────────────────────────────────
# verify_report -- the three required classification cases + failure cases
# ─────────────────────────────────────────────────────────────────────────

class TestVerifyReportClassifications(unittest.TestCase):
    """Explicitly covers Supported, Unsupported, and Partially Supported,
    per the Stage 3 testing requirement."""

    def test_supported_claim(self):
        report = "IBM announced a new 1000 qubit quantum processor in 2025."
        sources = "IBM has announced plans for a new 1000 qubit quantum processor to launch in 2025."
        result = grounding.verify_report(report, sources)
        self.assertEqual(result["counts"]["supported"], 1)
        self.assertEqual(result["claims"][0]["classification"], "Supported")

    def test_unsupported_claim(self):
        report = "Dinosaurs went extinct 66 million years ago due to an asteroid impact."
        sources = "IBM has announced plans for a new 1000 qubit quantum processor to launch in 2025."
        result = grounding.verify_report(report, sources)
        self.assertEqual(result["counts"]["unsupported"], 1)
        self.assertEqual(result["claims"][0]["overlap_ratio"], 0.0)

    def test_partially_supported_claim(self):
        report = "The field continues to attract billions of dollars in investment."
        sources = "Investment in quantum computing startups reached several billion dollars in 2024."
        result = grounding.verify_report(report, sources)
        self.assertEqual(result["claims"][0]["classification"], "Partially Supported")

    def test_grounding_score_formula(self):
        # 2 supported, 1 partial, 1 unsupported -> (2*1.0 + 1*0.5)/4*100 = 62.5
        report = (
            "IBM announced a new 1000 qubit quantum processor in 2025.\n"
            "Google released the Willow chip in December 2024.\n"
            "The field continues to attract billions of dollars in investment.\n"
            "Dinosaurs went extinct 66 million years ago due to an asteroid impact.\n"
        )
        sources = (
            "IBM announced a new 1000 qubit quantum processor in 2025. "
            "Google released the Willow chip in December 2024. "
            "Investment in quantum computing reached several billion dollars in 2024."
        )
        result = grounding.verify_report(report, sources)
        self.assertEqual(result["counts"]["total"], 4)
        expected = round(
            (result["counts"]["supported"] * 1.0 + result["counts"]["partially_supported"] * 0.5)
            / result["counts"]["total"] * 100, 1
        )
        self.assertEqual(result["grounding_score"], expected)


class TestVerifyReportFailureCases(unittest.TestCase):

    def test_no_checkable_claims_returns_none_not_zero(self):
        result = grounding.verify_report("# Title\n## Section", "some sources")
        self.assertIsNone(result["grounding_score"])
        self.assertEqual(result["counts"]["total"], 0)

    def test_empty_sources_does_not_crash_and_is_genuinely_zero(self):
        result = grounding.verify_report(
            "This is a genuinely long factual claim about something specific.", ""
        )
        self.assertEqual(result["grounding_score"], 0.0)
        self.assertEqual(result["counts"]["unsupported"], 1)

    def test_empty_report_and_empty_sources(self):
        result = grounding.verify_report("", "")
        self.assertIsNone(result["grounding_score"])

    def test_claim_made_entirely_of_stopwords_is_skipped_not_crashed(self):
        # "this is that and this is not" -- 7 words, passes MIN_CLAIM_WORDS,
        # but every word is a stopword, so claim_words is empty after
        # tokenizing. Must be skipped, not raise a ZeroDivisionError.
        result = grounding.verify_report("this is that and this is not there.", "some sources here")
        self.assertEqual(result["counts"]["total"], 0)
        self.assertIsNone(result["grounding_score"])


class TestFormatGroundingReport(unittest.TestCase):

    def test_na_case(self):
        result = grounding.verify_report("", "sources")
        text = grounding.format_grounding_report(result)
        self.assertIn("N/A", text)

    def test_unsupported_claims_are_listed(self):
        report = "Dinosaurs went extinct 66 million years ago due to an asteroid impact."
        result = grounding.verify_report(report, "totally unrelated source text about cooking recipes")
        text = grounding.format_grounding_report(result)
        self.assertIn("Unsupported Claims", text)
        self.assertIn("Dinosaurs", text)

    def test_all_supported_case_has_no_warning_section(self):
        report = "IBM announced a new processor."
        sources = "IBM announced a new processor this year."
        result = grounding.verify_report(report, sources)
        text = grounding.format_grounding_report(result)
        self.assertNotIn("Unsupported Claims", text)


# ─────────────────────────────────────────────────────────────────────────
# Integration: qa_agent (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    async def ainvoke(self, messages, config=None):
        return FakeResponse("fake qa verdict text")


def fake_get_llm():
    return FakeLLM()


class TestQaAgentIntegration(unittest.TestCase):

    def test_qa_agent_adds_grounding_fields_without_breaking_qa_review(self):
        state = {
            "query": "test",
            "research_data": "IBM announced a new 1000 qubit quantum processor in 2025.",
            "extra_context": "Investment in quantum computing reached several billion dollars in 2024.",
            "report": (
                "# Report\n## Findings\n"
                "IBM announced a new 1000 qubit quantum processor in 2025.\n"
                "Dinosaurs went extinct 66 million years ago due to an asteroid.\n"
            ),
            "analysis": "", "qa_review": "", "messages": [], "error": "",
        }
        with patch.object(agents, "get_llm", fake_get_llm):
            result = asyncio.run(agents.qa_agent(state))

        self.assertEqual(result["qa_review"], "fake qa verdict text")  # unchanged behavior
        self.assertIn("grounding_score", result)
        self.assertIn("grounding_report", result)
        self.assertIsNotNone(result["grounding_score"])
        self.assertIn("Dinosaurs", result["grounding_report"])  # unsupported claim flagged
        self.assertEqual(len(result["messages"]), 2)  # QA message + grounding message


# ─────────────────────────────────────────────────────────────────────────
# Integration: full compiled graph (mocked)
# ─────────────────────────────────────────────────────────────────────────

class TestFullGraphIntegration(unittest.TestCase):

    def test_grounding_fields_survive_the_full_graph(self):
        """Confirms grounding_report/grounding_score are declared in
        state.py's AgentState -- LangGraph silently drops any state key
        NOT declared in the TypedDict schema (verified empirically before
        writing this code), so this test would fail loudly if that
        declaration were ever accidentally removed."""

        def fake_search(query, max_results=5):
            # Stage 4: agents.py calls search_web_ranked() (dict-returning),
            # not search_web() (string-returning) -- patching search_web here
            # would be a silent no-op (the real network-backed function would
            # run instead). Must match the real function's contract.
            return {
                "formatted": "[1] Title\nIBM announced a new 1000 qubit quantum processor in 2025.\nSource: http://x.com",
                "sources": [{
                    "title": "Title", "url": "http://x.com",
                    "content": "IBM announced a new 1000 qubit quantum processor in 2025.",
                    "relevance": 0.5, "domain_trust": 0.5, "richness": 0.0,
                    "quality_score": 0.5, "quality_tier": "medium",
                }],
                "avg_quality": 0.5,
            }

        with patch.object(agents, "get_llm", fake_get_llm), \
             patch.object(agents, "search_web_ranked", fake_search):
            result = graph.run_research("test query")

        self.assertIn("grounding_score", result)
        self.assertIn("grounding_report", result)
        self.assertEqual(result["error"], "")

    def test_mocked_full_pipeline_with_one_real_and_one_hallucinated_claim(self):
        """A more realistic mocked end-to-end run: the writer produces a
        report with one claim genuinely backed by research_data/extra_context
        and one fabricated claim with no basis in either. Demonstrates the
        feature behaves sensibly on a full pipeline run, not just in
        isolated unit tests."""

        call_count = {"n": 0}

        class ScriptedLLM:
            async def ainvoke(self, messages, config=None):
                call_count["n"] += 1
                # Call order: 1st = research_agent, 2nd & 3rd = analyst +
                # extra_search (concurrent, order between these two isn't
                # guaranteed, but both come after call 1 and before call 4
                # since parallel_step only starts after research_agent
                # returns), 4th = writer, 5th = qa. research_agent's output
                # becomes state["research_data"], which IS one of the two
                # fields grounding checks against -- so it must actually
                # contain the "real" fact for this test's claim to have
                # anything to match. (An earlier version of this test
                # forgot this and got a 0% score for the wrong reason --
                # caught by inspecting the actual research_data/extra_context
                # the mock produced, not by assuming the mock was right.)
                if call_count["n"] == 1:
                    return FakeResponse("IBM announced a new 1000 qubit quantum processor in 2025.")
                if call_count["n"] == 4:
                    return FakeResponse(
                        "# Report\n## Findings\n"
                        "IBM announced a new 1000 qubit quantum processor in 2025.\n"
                        "The Eiffel Tower was originally intended to be dismantled after 20 years.\n"
                    )
                return FakeResponse("generic filler content about the topic")

        def scripted_get_llm():
            return ScriptedLLM()

        def fake_search(query, max_results=5):
            # Stage 4: same fix as above -- must match search_web_ranked()'s
            # dict contract, not search_web()'s plain-string one.
            return {
                "formatted": "[1] Title\nIBM announced a new 1000 qubit quantum processor in 2025.\nSource: http://x.com",
                "sources": [{
                    "title": "Title", "url": "http://x.com",
                    "content": "IBM announced a new 1000 qubit quantum processor in 2025.",
                    "relevance": 0.5, "domain_trust": 0.5, "richness": 0.0,
                    "quality_score": 0.5, "quality_tier": "medium",
                }],
                "avg_quality": 0.5,
            }

        with patch.object(agents, "get_llm", scripted_get_llm), \
             patch.object(agents, "search_web_ranked", fake_search):
            result = graph.run_research("quantum computing")

        # state only exposes grounding_score (float) and grounding_report
        # (a formatted string) -- not the raw counts dict, which is
        # internal to agents.qa_agent. So the 1-supported/1-unsupported
        # split is checked via the formula (2 claims, 1 fully backed by
        # research_data, 1 with zero overlap -> (1*1.0 + 0*0.5)/2*100 = 50.0)
        # and via what actually appears in the rendered grounding_report text.
        self.assertEqual(result["grounding_score"], 50.0)
        self.assertIn("Eiffel Tower", result["grounding_report"])  # hallucinated claim flagged
        self.assertNotIn("Unsupported Claims (2)", result["grounding_report"])  # only 1 of 2 is unsupported, not both
        self.assertIn("1 unsupported", result["grounding_report"])


# ─────────────────────────────────────────────────────────────────────────
# Database migration
# ─────────────────────────────────────────────────────────────────────────

class TestDatabaseMigration(unittest.TestCase):

    def setUp(self):
        self.db_path = database.DB_PATH
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_old_schema_database_migrates_without_data_loss(self):
        # Manually create an OLD-schema database (pre-Stage-3, no grounding columns)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                research_data TEXT,
                analysis TEXT,
                report TEXT,
                qa_review TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        c.execute(
            'INSERT INTO sessions (query, research_data, analysis, report, qa_review, created_at) '
            'VALUES (?,?,?,?,?,?)',
            ('old query', 'old research', 'old analysis', 'old report', 'old qa', '2026-01-01 00:00')
        )
        conn.commit()
        conn.close()

        database.init_db()  # the NEW init_db(), run against the OLD-schema db

        rows = database.get_all_sessions()
        self.assertEqual(len(rows), 1)
        full_row = database.get_session_by_id(rows[0][0])
        self.assertEqual(full_row[4], 'old report')  # untouched original data
        self.assertEqual(full_row[7], '')  # grounding_report defaulted
        self.assertIsNone(full_row[8])  # grounding_score defaulted

    def test_new_session_round_trips_grounding_data(self):
        database.init_db()
        new_id = database.save_session(
            'new query', 'new research', 'new analysis', 'new report', 'new qa',
            grounding_report='**Grounding Score: 75.0%**', grounding_score=75.0
        )
        row = database.get_session_by_id(new_id)
        self.assertEqual(row[7], '**Grounding Score: 75.0%**')
        self.assertEqual(row[8], 75.0)

    def test_save_session_without_grounding_args_still_works(self):
        """Backward compatibility: any caller that doesn't pass the new
        kwargs (there are none left in this codebase, but this guards
        against a future one) must not break."""
        database.init_db()
        new_id = database.save_session('q', 'r', 'a', 'rep', 'qa')
        row = database.get_session_by_id(new_id)
        self.assertEqual(row[7], '')
        self.assertIsNone(row[8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
