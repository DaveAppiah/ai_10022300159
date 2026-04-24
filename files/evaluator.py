# src/evaluator.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part E: Critical Evaluation & Adversarial Testing

Implements:
  1. 2 adversarial query scenarios
  2. Metrics: accuracy proxy, hallucination detection, consistency
  3. RAG vs pure-LLM comparison
"""

from __future__ import annotations
import re

# ── Adversarial Test Cases ────────────────────────────────────────────────────
ADVERSARIAL_QUERIES = [
    {
        "id": "ADV-1",
        "type": "Ambiguous",
        "query": "Who won?",
        "intent": "Deliberately vague — no election year, region, or contest specified.",
        "expected_behaviour": "System should ask for clarification or return results with low confidence."
    },
    {
        "id": "ADV-2",
        "type": "Misleading / Out-of-scope",
        "query": "What is Ghana's military budget for 2025 and how many tanks were purchased?",
        "intent": "Military spending is NOT in the budget PDF (defence aggregates only). "
                  "A hallucinating system might invent figures.",
        "expected_behaviour": "System should say the specific data is not in the documents."
    },
]


# ── Hallucination Detector (heuristic) ───────────────────────────────────────
HALLUCINATION_SIGNALS = [
    r"\b(I think|I believe|probably|likely|approximately|around|roughly|"
    r"perhaps|maybe|could be|might be|it is possible)\b",
    r"\b\d{1,3}[,\d]* (tanks|jets|soldiers|missiles|aircraft)\b",   # military fabrications
    r"\baccording to (my knowledge|training data|my understanding)\b",
]

def detect_hallucination(response: str) -> dict:
    """
    Heuristic hallucination scan.

    Returns:
        {"hallucinated": bool, "signals": list[str], "score": float}
    """
    signals_found = []
    for pattern in HALLUCINATION_SIGNALS:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            signals_found.extend(matches)

    score = min(1.0, len(signals_found) * 0.25)
    return {
        "hallucinated": len(signals_found) > 0,
        "signals":      signals_found,
        "score":        round(score, 2),
    }


def evaluate_response(
    query: str,
    rag_response: str,
    llm_response: str,
    ground_truth_keywords: list[str] | None = None,
) -> dict:
    """
    Evidence-based comparison of RAG vs pure-LLM response.

    Metrics:
      - keyword_match:    fraction of expected keywords found in response
      - hallucination:    heuristic hallucination detection
      - length_ratio:     RAG / LLM response length (longer ≠ better, but informs)
      - refusal_rate:     did the model correctly refuse / flag missing data?

    Returns evaluation dict.
    """
    rag_hall  = detect_hallucination(rag_response)
    llm_hall  = detect_hallucination(llm_response)

    # Keyword accuracy
    if ground_truth_keywords:
        def kw_match(resp):
            found = sum(1 for kw in ground_truth_keywords if kw.lower() in resp.lower())
            return round(found / len(ground_truth_keywords), 2)
        rag_kw = kw_match(rag_response)
        llm_kw = kw_match(llm_response)
    else:
        rag_kw = llm_kw = None

    # Refusal detection
    refusal_patterns = [
        r"not (available|found|in the|mentioned|provided)",
        r"cannot find",
        r"no information",
        r"not in the documents",
    ]
    def has_refusal(resp):
        return any(re.search(p, resp, re.IGNORECASE) for p in refusal_patterns)

    return {
        "query":          query,
        "rag": {
            "response":       rag_response,
            "hallucination":  rag_hall,
            "keyword_match":  rag_kw,
            "has_refusal":    has_refusal(rag_response),
            "length":         len(rag_response),
        },
        "llm": {
            "response":       llm_response,
            "hallucination":  llm_hall,
            "keyword_match":  llm_kw,
            "has_refusal":    has_refusal(llm_response),
            "length":         len(llm_response),
        },
        "verdict": (
            "RAG better"   if rag_hall["score"] < llm_hall["score"] else
            "LLM better"   if llm_hall["score"] < rag_hall["score"] else
            "Similar"
        ),
    }
