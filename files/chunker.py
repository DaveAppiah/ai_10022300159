# src/chunker.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part A: Chunking Strategy Design & Implementation

Two strategies are implemented:

1. FIXED-SIZE CHUNKING (for Budget PDF)
   - Chunk size: 512 tokens (≈ 2000 characters)
   - Overlap:    64 tokens  (≈ 256 characters)
   Justification:
     The budget PDF is long-form prose. Fixed-size chunks with overlap
     preserve sentence context at boundaries and keep chunks within the
     LLM's context window.  Overlap prevents losing information that
     spans two consecutive chunks.

2. ROW-BASED CHUNKING (for Election CSV)
   - Each CSV row becomes one chunk.
   Justification:
     Each election row is a self-contained fact (region, candidate, votes).
     Splitting rows would destroy relational meaning. No overlap is needed.

Comparative analysis is logged to logs/experiment_logs.md.
"""

from __future__ import annotations
import re
import uuid
from typing import Literal


# ── Constants ─────────────────────────────────────────────────────────────────
CHARS_PER_TOKEN = 4          # rough approximation
FIXED_CHUNK_CHARS = 512 * CHARS_PER_TOKEN   # 2048 chars
OVERLAP_CHARS    = 64  * CHARS_PER_TOKEN    # 256 chars


# ── Core chunking functions ───────────────────────────────────────────────────

def chunk_fixed_size(text: str, chunk_size: int = FIXED_CHUNK_CHARS,
                     overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Split *text* into fixed-size overlapping windows (character-level).

    Args:
        text:       Raw text to chunk.
        chunk_size: Max characters per chunk.
        overlap:    Overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    if not text.strip():
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_by_sentence(text: str, max_chars: int = FIXED_CHUNK_CHARS,
                      overlap_sentences: int = 2) -> list[str]:
    """
    Alternative strategy: sentence-boundary chunking.
    Accumulates sentences until max_chars is reached, then starts a new chunk.
    The last `overlap_sentences` sentences carry over to the next chunk.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            # carry-over for overlap
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Document-level chunker ────────────────────────────────────────────────────

def chunk_documents(raw_docs: list[dict],
                    strategy: Literal["fixed", "sentence"] = "fixed"
                    ) -> list[dict]:
    """
    Takes raw document dicts (from data_loader) and returns chunk dicts.

    For election CSV rows (source == "election_csv"):
        Each row is already one chunk — no further splitting needed.

    For budget PDF (source == "budget_pdf"):
        Apply the chosen text-splitting strategy.

    Returns list of chunk dicts:
        {
            "chunk_id": str,
            "text":     str,
            "source":   str,
            "metadata": dict,
            "chunk_index": int,
        }
    """
    all_chunks: list[dict] = []

    for doc in raw_docs:
        source = doc.get("source", "unknown")
        text   = doc.get("text", "")
        meta   = doc.get("metadata", {})

        if source == "election_csv":
            # Row-based: entire row text is one chunk
            chunk = _make_chunk(text, source, meta, 0)
            all_chunks.append(chunk)

        elif source == "budget_pdf":
            if strategy == "sentence":
                splits = chunk_by_sentence(text)
            else:
                splits = chunk_fixed_size(text)

            for idx, split_text in enumerate(splits):
                chunk = _make_chunk(split_text, source, meta, idx)
                all_chunks.append(chunk)

        else:
            # Generic fallback
            chunk = _make_chunk(text, source, meta, 0)
            all_chunks.append(chunk)

    print(f"[chunker] {len(all_chunks)} total chunks created "
          f"(strategy='{strategy}').")
    return all_chunks


def _make_chunk(text: str, source: str, metadata: dict, index: int) -> dict:
    return {
        "chunk_id":    str(uuid.uuid4()),
        "text":        text,
        "source":      source,
        "metadata":    metadata,
        "chunk_index": index,
    }


# ── Comparative analysis helper ───────────────────────────────────────────────

def compare_chunking_strategies(pdf_text: str) -> dict:
    """
    Returns stats for both strategies on the PDF text.
    Used in experiment logs and Part A analysis.
    """
    fixed_chunks    = chunk_fixed_size(pdf_text)
    sentence_chunks = chunk_by_sentence(pdf_text)

    def stats(chunks: list[str]) -> dict:
        lengths = [len(c) for c in chunks]
        return {
            "count": len(chunks),
            "avg_chars": round(sum(lengths) / max(len(lengths), 1)),
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }

    return {
        "fixed_size":  stats(fixed_chunks),
        "sentence":    stats(sentence_chunks),
    }
