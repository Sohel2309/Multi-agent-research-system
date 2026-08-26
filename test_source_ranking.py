"""
Unit tests for Stage 4: source_ranking.py.

Every quality_score assertion below is checked against arithmetic worked
by hand in the docstrings -- see source_ranking.py's module docstring for
the formula. These tests never call an LLM, Tavily, or the network.

Run with: python3 test_source_ranking.py
"""
import unittest

from source_ranking import (
    score_source,
    rank_sources,
    average_quality,
    classify_quality,
    format_ranked_results,
    format_quality_report,
    _domain_of,
    _domain_trust_score,
)


class TestDomainTrust(unittest.TestCase):

    def test_gov_domain_is_fully_trusted(self):
        self.assertEqual(_domain_trust_score("https://www.cdc.gov/flu"), 1.0)

    def test_edu_domain_is_fully_trusted(self):
        self.assertEqual(_domain_trust_score("https://mit.edu/research"), 1.0)

    def test_curated_high_trust_domain(self):
        self.assertEqual(_domain_trust_score("https://www.nature.com/articles/x"), 1.0)

    def test_org_domain_gets_partial_trust(self):
        self.assertEqual(_domain_trust_score("https://example.org/page"), 0.7)

    def test_unknown_commercial_domain_gets_baseline(self):
        self.assertEqual(_domain_trust_score("https://randomblog.com/post"), 0.5)

    def test_missing_url_is_penalized_not_neutral(self):
        """No URL at all means we can't verify the domain -- this should
        score BELOW the unknown-commercial baseline (0.5), not at it."""
        self.assertEqual(_domain_trust_score(""), 0.3)

    def test_www_prefix_is_stripped(self):
        self.assertEqual(_domain_of("https://www.reuters.com/x"), "reuters.com")

    def test_malformed_url_does_not_raise(self):
        # urlparse is lenient and returns an empty netloc for a string
        # with no scheme/host -- treated the same as "no URL" (0.3), not
        # a crash and not a false-neutral 0.5.
        self.assertEqual(_domain_trust_score("not a url at all"), 0.3)


class TestScoreSource(unittest.TestCase):

    def test_weak_source_hand_verified(self):
        """Low relevance, unknown domain, thin content with no numbers.
        Hand arithmetic: relevance=0.3, domain=0.5, richness=0.7*(11/500)=0.0154
        quality = 0.5*0.3 + 0.3*0.5 + 0.2*0.0154 = 0.303 (rounded)."""
        result = {
            "title": "Random Blog Post",
            "url": "https://randomblog.com/post",
            "content": "Short text.",
            "score": 0.3,
        }
        scored = score_source(result)
        self.assertEqual(scored["relevance"], 0.3)
        self.assertEqual(scored["domain_trust"], 0.5)
        self.assertEqual(scored["quality_score"], 0.303)
        self.assertEqual(scored["quality_tier"], "Low")

    def test_strong_source_hand_verified(self):
        """High relevance, curated-trusted domain, number-dense content.
        Hand arithmetic: relevance=0.9, domain=1.0,
        richness=0.7*(68/500) + 0.3*1 = 0.0952+0.3=0.3952 -> 0.392 rounded
        quality = 0.5*0.9 + 0.3*1.0 + 0.2*0.392 = 0.828."""
        result = {
            "title": "Nature Study",
            "url": "https://www.nature.com/articles/xyz",
            "content": "Study of 500 patients found a 42% reduction in risk over 10 years.",
            "score": 0.9,
        }
        scored = score_source(result)
        self.assertEqual(scored["relevance"], 0.9)
        self.assertEqual(scored["domain_trust"], 1.0)
        self.assertEqual(scored["quality_score"], 0.828)
        self.assertEqual(scored["quality_tier"], "High")

    def test_missing_score_field_defaults_to_neutral_relevance(self):
        result = {"title": "T", "url": "https://example.com/x", "content": "some content"}
        scored = score_source(result)
        self.assertEqual(scored["relevance"], 0.5)

    def test_out_of_range_score_falls_back_to_neutral(self):
        result = {"title": "T", "url": "https://example.com/x", "content": "c", "score": 5.0}
        scored = score_source(result)
        self.assertEqual(scored["relevance"], 0.5)

    def test_missing_url_and_content_does_not_raise(self):
        result = {"title": "T"}
        scored = score_source(result)
        self.assertEqual(scored["url"], "")
        self.assertEqual(scored["content"], "")
        self.assertEqual(scored["richness"], 0.0)

    def test_missing_title_gets_placeholder(self):
        result = {"url": "https://example.com", "content": "x"}
        scored = score_source(result)
        self.assertEqual(scored["title"], "(no title)")

    def test_quality_score_is_clamped_to_zero_one(self):
        result = {"url": "https://cdc.gov/x", "content": "x" * 1000, "score": 1.0}
        scored = score_source(result)
        self.assertLessEqual(scored["quality_score"], 1.0)
        self.assertGreaterEqual(scored["quality_score"], 0.0)


class TestClassifyQuality(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(classify_quality(0.75), "High")
        self.assertEqual(classify_quality(0.749), "Medium")
        self.assertEqual(classify_quality(0.5), "Medium")
        self.assertEqual(classify_quality(0.499), "Low")
        self.assertEqual(classify_quality(1.0), "High")
        self.assertEqual(classify_quality(0.0), "Low")


class TestRankSources(unittest.TestCase):

    def test_empty_list_returns_empty(self):
        self.assertEqual(rank_sources([]), [])
        self.assertEqual(rank_sources(None), [])

    def test_sorted_best_first(self):
        weak = {"title": "Weak", "url": "https://randomblog.com/a", "content": "Short text.", "score": 0.3}
        strong = {
            "title": "Strong",
            "url": "https://www.nature.com/b",
            "content": "Study of 500 patients found a 42% reduction in risk over 10 years.",
            "score": 0.9,
        }
        ranked = rank_sources([weak, strong])
        self.assertEqual(ranked[0]["title"], "Strong")
        self.assertEqual(ranked[1]["title"], "Weak")

    def test_duplicate_urls_deduplicated_first_wins(self):
        first = {"title": "First", "url": "https://x.com/a", "content": "c1", "score": 0.5}
        dup = {"title": "Duplicate", "url": "https://x.com/a", "content": "c2", "score": 0.99}
        ranked = rank_sources([first, dup])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["title"], "First")

    def test_results_with_no_url_are_never_deduplicated_against_each_other(self):
        a = {"title": "A", "url": "", "content": "c1", "score": 0.5}
        b = {"title": "B", "url": "", "content": "c2", "score": 0.5}
        ranked = rank_sources([a, b])
        self.assertEqual(len(ranked), 2)

    def test_non_dict_entries_are_skipped_not_raised(self):
        good = {"title": "Good", "url": "https://x.com", "content": "c", "score": 0.5}
        ranked = rank_sources([good, "not a dict", None, 42])
        self.assertEqual(len(ranked), 1)


class TestAverageQuality(unittest.TestCase):

    def test_empty_returns_none_not_zero(self):
        """Must never fabricate an average of 0 for zero sources -- same
        convention as grounding.verify_report's grounding_score=None."""
        self.assertIsNone(average_quality([]))

    def test_average_is_mean_of_quality_scores(self):
        ranked = rank_sources([
            {"title": "A", "url": "https://x.com/a", "content": "x" * 500, "score": 1.0},
            {"title": "B", "url": "https://y.com/b", "content": "", "score": 0.0},
        ])
        # A: relevance=1.0, domain=0.5, richness=0.7*1+0.3*0=0.7 -> 0.5+0.15+0.14=0.79
        # B: relevance=0.0, domain=0.5, richness=0.0 -> 0+0.15+0=0.15
        # avg = (0.79+0.15)/2 = 0.47
        self.assertEqual(average_quality(ranked), 0.47)


class TestFormatting(unittest.TestCase):

    def test_format_ranked_results_empty(self):
        self.assertEqual(format_ranked_results([]), "No results found.")

    def test_format_ranked_results_contains_quality_tag(self):
        ranked = rank_sources([
            {"title": "T", "url": "https://cdc.gov/x", "content": "some real content with numbers 42", "score": 0.9}
        ])
        text = format_ranked_results(ranked)
        self.assertIn("quality", text)
        self.assertIn("Source: https://cdc.gov/x", text)

    def test_format_quality_report_no_sources(self):
        text = format_quality_report([])
        self.assertIn("No sources were retrieved", text)

    def test_format_quality_report_includes_avg_and_counts(self):
        ranked = rank_sources([
            {"title": "T", "url": "https://cdc.gov/x", "content": "c" * 500 + " 42", "score": 0.9},
        ])
        text = format_quality_report(ranked)
        self.assertIn("avg quality", text)
        self.assertIn("1 High", text)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)
