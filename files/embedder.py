# src/embedder.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part B: Custom Embedding Pipeline

Uses sentence-transformers (all-MiniLM-L6-v2) — lightweight, fast, and
free. No API key required. Embeddings are 384-dimensional float32 vectors.

Design decision:
  all-MiniLM-L6-v2 was chosen because:
  - Strong performance on semantic similarity benchmarks
  - 384-dim output (balanced between quality and storage cost)
  - Fast inference (~14,000 sentences/sec on CPU)
  - Freely available, no rate limits
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model singleton ───────────────────────────────────────────────────────────
_MODEL: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _MODEL
    if _MODEL is None:
        print(f"[embedder] Loading model '{MODEL_NAME}' …")
        _MODEL = SentenceTransformer(MODEL_NAME)
        print("[embedder] Model ready.")
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings into a (N, 384) float32 numpy array.

    Args:
        texts:      List of strings to embed.
        batch_size: Batch size for the transformer forward pass.

    Returns:
        numpy array of shape (len(texts), 384), L2-normalised (unit vectors).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,   # produces unit vectors → cosine sim = dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy array of shape (1, 384).
    """
    return embed_texts([query])
