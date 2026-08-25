"""
Stage 3: Deterministic grounded fact-checking.

This does NOT call an LLM and does NOT use embeddings or a vector store --
on purpose. The existing qa_agent (agents.py) already asks an LLM to grade
its own report against its own sources, which is exactly the "vibes-based,
self-graded" weakness this stage exists to add a check against. Adding a
SECOND LLM call that also grades itself would not fix that weakness, so
this module instead does simple, deterministic lexical-overlap matching
between each claim in the report and the raw text the system already
retrieved (research_data + extra_context) -- no new search, no new
external calls, nothing beyond the Python standard library `re` module.

THE FORMULA (this is the whole algorithm -- defend this in an interview):
  1. Split the final report into individual claims (roughly: sentences,
     after stripping markdown headers/bullets/bold markers).
  2. For each claim, tokenize it into lowercase, punctuation-stripped,
     stopword-filtered "significant words".
  3. Tokenize the combined source text (research_data + extra_context) the
     same way.
  4. overlap_ratio = |claim's significant words that also appear in the
     source text| / |claim's significant words|
  5. Classify: overlap_ratio >= 0.6 -> Supported
               0.3 <= overlap_ratio < 0.6 -> Partially Supported
               overlap_ratio < 0.3 -> Unsupported
  6. Grounding Score = (supported*1.0 + partially_supported*0.5) / total_claims * 100

KNOWN, NAMED LIMITATION (say this out loud, don't hide it): this measures
lexical overlap, not semantic truth or entailment. A claim can share many
words with the source text and still misstate what the source says (e.g.
flipping a number or a comparison), and a claim can be true but phrased
with entirely different words than the source and get flagged as
"Unsupported." This is a simple, transparent heuristic appropriate for a
student project -- not a claim of semantic fact-checking. The 0.6/0.3
thresholds are a starting point calibrated by hand-inspection of a few
example reports (see test_grounding.py), not tuned against a large
labeled dataset -- there wasn't one to tune against.
"""

import re

SUPPORTED_THRESHOLD = 0.6
PARTIAL_THRESHOLD = 0.3

MIN_CLAIM_WORDS = 5  # shorter fragments are usually stray bullet labels, not real claims

# Small, hardcoded stopword list (no NLTK dependency -- keeps this simple
# and dependency-free, per the "no unnecessary frameworks" constraint).
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "and", "or",
    "but", "if", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "now", "also", "its",
    "their", "our", "your", "as", "which", "who", "whom", "what",
})


def _clean_markdown(text: str) -> str:
    """Strips markdown structure (headers, bold/italic, bullet/numbered
    list markers) while leaving the underlying words intact, so headers
    like '## Key Findings' don't get treated as factual claims."""
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^[\-\*\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


def extract_claims(report_text: str, min_words: int = MIN_CLAIM_WORDS) -> list:
    """Splits a report into candidate factual claims.

    This is a simple heuristic splitter (line-by-line, then sentence
    boundaries within each line), not a proper NLP sentence tokenizer.
    It deliberately avoids splitting on a period immediately followed or
    preceded by a digit, so "3.5%" or "the U.S." don't get chopped
    mid-token. Lines/fragments shorter than `min_words` words are dropped
    (they're almost always bullet labels or section fragments, not
    checkable claims).
    """
    if not report_text or not report_text.strip():
        return []

    cleaned = _clean_markdown(report_text)
    claims = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences = re.split(r"(?<!\d)(?<=[.!?])\s+(?!\d)", line)
        for s in sentences:
            s = s.strip(" -\u2022")
            if not s:
                continue
            if len(s.split()) < min_words:
                continue
            claims.append(s)
    return claims


def _tokenize(text: str) -> set:
    """Lowercase, punctuation-stripped, stopword-filtered word set.
    Keeps digits and '%' as part of tokens (e.g. "51%" stays "51%") since
    numbers are often the most important part of a factual claim."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9%]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def classify_overlap(overlap_ratio: float) -> str:
    if overlap_ratio >= SUPPORTED_THRESHOLD:
        return "Supported"
    elif overlap_ratio >= PARTIAL_THRESHOLD:
        return "Partially Supported"
    else:
        return "Unsupported"


def verify_report(report_text: str, sources_text: str) -> dict:
    """Runs the full grounding check described in this module's docstring.

    Returns:
        {
          "claims": [{"claim": str, "overlap_ratio": float, "classification": str}, ...],
          "counts": {"supported": int, "partially_supported": int, "unsupported": int, "total": int},
          "grounding_score": float | None,  # None if there were no checkable claims -- never fabricated as 0
        }
    """
    claims = extract_claims(report_text)
    source_words = _tokenize(sources_text)

    claim_results = []
    for claim in claims:
        claim_words = _tokenize(claim)
        if not claim_words:
            # every word in this "claim" was a stopword/too short -- not
            # something we can meaningfully check, so skip rather than
            # force a 0/0 division or a fabricated classification.
            continue
        overlap_ratio = len(claim_words & source_words) / len(claim_words)
        claim_results.append({
            "claim": claim,
            "overlap_ratio": round(overlap_ratio, 3),
            "classification": classify_overlap(overlap_ratio),
        })

    total = len(claim_results)
    supported = sum(1 for c in claim_results if c["classification"] == "Supported")
    partial = sum(1 for c in claim_results if c["classification"] == "Partially Supported")
    unsupported = sum(1 for c in claim_results if c["classification"] == "Unsupported")

    if total == 0:
        grounding_score = None
    else:
        grounding_score = round((supported * 1.0 + partial * 0.5) / total * 100, 1)

    return {
        "claims": claim_results,
        "counts": {
            "supported": supported,
            "partially_supported": partial,
            "unsupported": unsupported,
            "total": total,
        },
        "grounding_score": grounding_score,
    }


def format_grounding_report(result: dict) -> str:
    """Formats a verify_report() result as a Markdown string for display
    in the Streamlit UI and for storage alongside qa_review."""
    counts = result["counts"]
    score = result["grounding_score"]

    if score is None:
        lines = [
            "**Grounding Score:** N/A -- no checkable claims were extracted from this report "
            "(it may be very short, or made up entirely of headers/short fragments).",
        ]
        return "\n".join(lines)

    lines = [
        f"**Grounding Score: {score}%** "
        f"({counts['supported']} supported, {counts['partially_supported']} partially supported, "
        f"{counts['unsupported']} unsupported -- out of {counts['total']} claims checked)",
        "",
        "_Formula: (supported x 1.0 + partially_supported x 0.5) / total_claims x 100. "
        "This measures word-overlap with this system's own retrieved sources, not independent "
        "fact verification -- see grounding.py for the full explanation and its limitations._",
        "",
    ]

    unsupported_claims = [c for c in result["claims"] if c["classification"] == "Unsupported"]
    if unsupported_claims:
        lines.append(f"**\u26a0\ufe0f Unsupported Claims ({len(unsupported_claims)}):**")
        for c in unsupported_claims:
            lines.append(f"- {c['claim']}  _(word overlap with sources: {c['overlap_ratio']*100:.0f}%)_")
        lines.append("")

    partial_claims = [c for c in result["claims"] if c["classification"] == "Partially Supported"]
    if partial_claims:
        lines.append(f"**\U0001f7e1 Partially Supported Claims ({len(partial_claims)}):**")
        for c in partial_claims:
            lines.append(f"- {c['claim']}  _(word overlap with sources: {c['overlap_ratio']*100:.0f}%)_")
        lines.append("")

    if not unsupported_claims and not partial_claims:
        lines.append("All checked claims were classified as Supported.")

    return "\n".join(lines)
