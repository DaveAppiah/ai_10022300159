# src/retriever.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part B: Custom Retrieval System

Features implemented:
  1. Top-K vector retrieval (via VectorStore)
  2. Similarity scoring (cosine, displayed in UI)
  3. HYBRID SEARCH — keyword (BM25) + vector fusion          ← extension choice
  4. Query Expansion                                          ← Innovation (Part G)

Failure case handling:
  - If top-1 score < RELEVANCE_THRESHOLD, the system flags "low confidence"
    and widens the search window before re-ranking.
"""

from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi

from embedder import embed_query
from vector_store import VectorStore

RELEVANCE_THRESHOLD = 0.30   # chunks below this are likely irrelevant
TOP_K_DEFAULT = 5


# ── Query Expansion ───────────────────────────────────────────────────────────
_EXPANSION_MAP: dict[str, list[str]] = {
    "election":     ["vote", "ballot", "constituency", "polling", "result"],
    "budget":       ["expenditure", "revenue", "fiscal", "allocation", "finance"],
    "ndc":          ["national democratic congress", "john mahama"],
    "npp":          ["new patriotic party", "akufo-addo", "bawumia"],
    "president":    ["head of state", "presidential candidate"],
    "gdp":          ["gross domestic product", "economic growth", "output"],
    "inflation":    ["price level", "cost of living", "cpi"],
    "education":    ["school", "university", "learning", "student"],
    "health":       ["hospital", "medical", "healthcare", "nhis"],
    "tax":          ["revenue", "irs", "levy", "duty", "vat"],
    "ghana":        ["republic of ghana", "gh"],
}

def expand_query(query: str) -> str:
    """
    Part G Innovation: Query Expansion.

    Appends synonyms / related terms to the raw query so that the embedding
    captures broader semantic intent.

    Example:
      "Who won the NDC election?" →
      "Who won the NDC election? national democratic congress john mahama
       vote ballot constituency polling result"
    """
    lower = query.lower()
    expansions: list[str] = []
    for keyword, synonyms in _EXPANSION_MAP.items():
        if keyword in lower:
            expansions.extend(synonyms)

    if expansions:
        # Deduplicate and append
        unique_exp = list(dict.fromkeys(expansions))
        expanded = query + " " + " ".join(unique_exp)
        return expanded
    return query


# ── BM25 Keyword Search ───────────────────────────────────────────────────────
class BM25Index:
    """Thin wrapper around rank-bm25 for keyword retrieval."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenised = [self._tokenise(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenised)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, k: int = TOP_K_DEFAULT) -> list[dict]:
        tokens = self._tokenise(query)
        scores = self.bm25.get_scores(tokens)
        top_k = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
        return results


# ── Hybrid Retriever ──────────────────────────────────────────────────────────
class Retriever:
    """
    Hybrid retrieval: vector search + BM25, combined with Reciprocal Rank Fusion.

    RRF score = Σ  1 / (rank_i + 60)   for each result list i
    """

    def __init__(self, vector_store: VectorStore, chunks: list[dict]):
        self.vector_store = vector_store
        self.bm25_index   = BM25Index(chunks)
        self.chunks        = chunks

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_DEFAULT,
        use_expansion: bool = True,
        alpha: float = 0.7,          # weight for vector vs bm25 (0 = pure BM25, 1 = pure vector)
    ) -> dict:
        """
        Main retrieval method.

        Args:
            query:          User query string.
            k:              Number of top chunks to return.
            use_expansion:  Whether to apply query expansion.
            alpha:          Vector weight in hybrid fusion.

        Returns:
            dict with keys:
              - 'results':      list of chunk dicts with 'final_score'
              - 'query_used':   the (possibly expanded) query
              - 'low_confidence': bool flag
              - 'retrieval_log': list of log strings
        """
        log: list[str] = []

        # 1. Query Expansion
        if use_expansion:
            expanded_query = expand_query(query)
            if expanded_query != query:
                log.append(f"Query expanded: '{query}' → '{expanded_query}'")
        else:
            expanded_query = query

        # 2. Vector retrieval
        q_vec = embed_query(expanded_query)
        vec_results = self.vector_store.search(q_vec, k=k * 2)   # over-fetch for fusion
        log.append(f"Vector retrieval: {len(vec_results)} candidates")

        # 3. BM25 retrieval
        bm25_results = self.bm25_index.search(expanded_query, k=k * 2)
        log.append(f"BM25 retrieval: {len(bm25_results)} candidates")

        # 4. Reciprocal Rank Fusion
        fused = self._rrf_fusion(vec_results, bm25_results, alpha=alpha)

        # 5. Take top-k
        top_results = fused[:k]

        # 6. Low-confidence detection (failure case handling)
        low_conf = False
        if top_results and top_results[0].get("similarity_score", 1.0) < RELEVANCE_THRESHOLD:
            low_conf = True
            log.append(
                f"⚠ Low confidence: top score {top_results[0].get('similarity_score', 0):.3f} "
                f"< threshold {RELEVANCE_THRESHOLD}"
            )
            # FIX: widen BM25 and re-fuse
            bm25_wide = self.bm25_index.search(query, k=k * 4)
            vec_wide  = self.vector_store.search(q_vec, k=k * 4)
            fused_wide = self._rrf_fusion(vec_wide, bm25_wide, alpha=0.5)
            top_results = fused_wide[:k]
            log.append("Fix applied: widened BM25 search, re-fused results.")

        log.append(f"Final: {len(top_results)} chunks returned.")

        return {
            "results":       top_results,
            "query_used":    expanded_query,
            "low_confidence": low_conf,
            "retrieval_log": log,
        }

    # ── Fusion helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _rrf_fusion(
        vec_results: list[dict],
        bm25_results: list[dict],
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion of two ranked lists.
        Returns sorted list with 'final_score' key.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        # Vector scores (weight = alpha)
        for rank, chunk in enumerate(vec_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + alpha * (1.0 / (rank + rrf_k))
            chunk_map[cid] = chunk

        # BM25 scores (weight = 1 - alpha)
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * (1.0 / (rank + rrf_k))
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for cid in sorted_ids:
            c = dict(chunk_map[cid])
            c["final_score"] = round(scores[cid], 6)
            result.append(c)

        return result
