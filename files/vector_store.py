# src/vector_store.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part B: Custom FAISS Vector Store

Manual implementation — no wrappers around FAISS.

Design:
  - IndexFlatIP (inner product) is equivalent to cosine similarity when
    vectors are unit-normalised (which we enforce in embedder.py).
  - All chunk metadata is stored in a parallel Python list so we can
    return full chunk dicts alongside scores.
"""

from __future__ import annotations
import numpy as np
import faiss

from embedder import embed_texts


class VectorStore:
    """
    Wraps a FAISS IndexFlatIP with a metadata store.

    Usage:
        store = VectorStore()
        store.add_chunks(chunk_list)          # build index
        results = store.search(query_vec, k=5)
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # cosine sim (vectors are unit-normed)
        self.chunks: list[dict] = []           # mirrors index row order

    # ── Build ─────────────────────────────────────────────────────────────────
    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and add them to the FAISS index.

        Args:
            chunks: List of chunk dicts (must have 'text' key).
        """
        if not chunks:
            print("[vector_store] No chunks to index.")
            return

        texts = [c["text"] for c in chunks]
        print(f"[vector_store] Embedding {len(texts)} chunks …")
        vectors = embed_texts(texts)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        print(f"[vector_store] Index now contains {self.index.ntotal} vectors.")

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query_vector: np.ndarray, k: int = 5
               ) -> list[dict]:
        """
        Retrieve top-k most similar chunks.

        Args:
            query_vector: Shape (1, dim) float32 unit vector.
            k:            Number of results to return.

        Returns:
            List of dicts: chunk dict + 'similarity_score' key.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])          # shallow copy
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        faiss.write_index(self.index, path + ".faiss")
        import pickle, pathlib
        pathlib.Path(path + ".meta").write_bytes(pickle.dumps(self.chunks))

    def load(self, path: str) -> None:
        import pickle, pathlib
        self.index = faiss.read_index(path + ".faiss")
        self.chunks = pickle.loads(pathlib.Path(path + ".meta").read_bytes())
        self.dim = self.index.d
        print(f"[vector_store] Loaded index with {self.index.ntotal} vectors.")

    @property
    def size(self) -> int:
        return self.index.ntotal
