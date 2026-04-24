# Architecture & System Design Documentation
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 — Introduction to Artificial Intelligence — 2026

---

## 1. System Overview

This document describes the design and implementation of a custom RAG (Retrieval-Augmented
Generation) chat assistant built for Academic City University. The system is built
entirely from scratch — no LangChain, LlamaIndex, or pre-built RAG pipelines.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                            │
│                                                                  │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │  Ghana Election CSV     │   │  Ghana 2025 Budget PDF      │  │
│  │  (GodwinDansoAcity/     │   │  (mofep.gov.gh)             │  │
│  │   acitydataset)         │   │                             │  │
│  └────────────┬────────────┘   └──────────────┬──────────────┘  │
│               │ load_election_csv()            │ load_budget_pdf()│
│               ▼                               ▼                  │
│         Row-based chunks              Fixed-size chunks          │
│         (1 row = 1 chunk)             (512 tok, 64 overlap)      │
│               │                               │                  │
│               └───────────────┬───────────────┘                  │
│                               ▼                                  │
│                    chunk_documents() → list[dict]                │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                        EMBEDDING & INDEXING                      │
│                                                                  │
│  embed_texts() → sentence-transformers all-MiniLM-L6-v2         │
│  384-dim float32 unit vectors (cosine sim = dot product)        │
│                                                                  │
│  VectorStore.add_chunks() → FAISS IndexFlatIP                   │
│  Saved to disk: data/rag_index.faiss + data/rag_index.meta      │
│                                                                  │
│  BM25Index built in parallel for keyword retrieval              │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼ (at query time)
┌──────────────────────────────────────────────────────────────────┐
│                        QUERY PROCESSING                          │
│                                                                  │
│  1. Query Expansion (Innovation / Part G)                        │
│     expand_query() appends domain synonyms                      │
│     e.g. "NDC" → "NDC national democratic congress john mahama…"│
│                                                                  │
│  2. Embedding                                                    │
│     embed_query() → (1, 384) unit vector                        │
│                                                                  │
│  3. Hybrid Retrieval                                             │
│     ┌────────────────────┐   ┌────────────────────┐            │
│     │  FAISS vector      │   │  BM25 keyword      │            │
│     │  search (top 2K)   │   │  search (top 2K)   │            │
│     └─────────┬──────────┘   └─────────┬──────────┘            │
│               │                         │                        │
│               └───────────┬─────────────┘                        │
│                           ▼                                      │
│              RRF Fusion (α=0.7 vector, 0.3 BM25)               │
│              → sorted list of chunks with final_score            │
│                                                                  │
│  4. Low-Confidence Failsafe                                      │
│     If top score < 0.30: widen search, re-fuse, flag UI         │
│                                                                  │
│  5. Top-K selection (default K=5)                                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        PROMPT CONSTRUCTION                       │
│                                                                  │
│  build_prompt(query, chunks, version)                           │
│  - Injects retrieved context into <context> XML block           │
│  - Truncates to max 6,000 chars                                 │
│  - Applies hallucination-control instructions                   │
│  - 3 template versions for A/B testing                          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        LLM GENERATION                            │
│                                                                  │
│  Anthropic Claude claude-sonnet-4 via official Python SDK       │
│  max_tokens=1024                                                 │
│  system prompt + user message (context + query)                 │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        UI / DISPLAY                              │
│                                                                  │
│  Streamlit app.py                                               │
│  - Chat tab: query input, answer, chunks, logs, prompt          │
│  - Adversarial tab: ADV-1, ADV-2 test runner + eval             │
│  - Compare tab: RAG vs pure LLM side-by-side                    │
│  - Architecture tab: this diagram + justifications              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 data_loader.py
Handles downloading and cleaning both datasets. PDF text is extracted using PyMuPDF,
which correctly handles multi-column layouts and avoids garbled table rendering.
CSV cleaning normalises numeric columns and fills missing string fields.

### 3.2 chunker.py
Two strategies:
- **Fixed-size (default):** 512-token equivalent windows with 64-token overlap.
  Overlap prevents losing context at chunk boundaries — critical for multi-sentence
  facts in the budget document.
- **Sentence-based:** accumulates sentences up to 512 tokens, then starts a new chunk
  with 2-sentence carry-over. More linguistically coherent but less predictable in size.

Election CSV rows use row-based chunking — each row is a self-contained fact.

### 3.3 embedder.py
Wraps `sentence-transformers` with L2-normalised output so that inner-product
in FAISS equals cosine similarity. Batch processing for efficiency.

### 3.4 vector_store.py
Manual FAISS IndexFlatIP construction. No HNSW approximation — exact search is
acceptable given corpus size (<5K chunks). Index is persisted to disk to avoid
re-embedding on every restart.

### 3.5 retriever.py
Hybrid retrieval combining:
- **Vector search** (semantic, ~70% weight)
- **BM25 keyword search** (lexical, ~30% weight)
- **Reciprocal Rank Fusion** — parameter-free combination method that handles
  the different score scales of cosine and BM25

Also implements **query expansion** (Part G Innovation) — prepends domain-specific
synonyms to the query before embedding. Highly effective for Ghanaian government
acronyms (NDC, NPP, NHIS, GES).

### 3.6 prompt_builder.py
Three prompt templates tested (Part C). All include:
- `<context>` XML block for clear context injection
- Explicit "only use context" instruction
- Refusal instruction when context is insufficient
- Source attribution request

### 3.7 rag_pipeline.py
Orchestrates all stages with a `PipelineLogger` injected at each step.
Supports `pure_llm_mode=True` for Part E RAG vs LLM comparison.

### 3.8 evaluator.py
Heuristic hallucination detection using regex patterns for hedging language
and domain-specific fabrication signals. Evidence-based RAG vs LLM comparison
using keyword match, refusal detection, and hallucination scoring.

---

## 4. Why This Design Suits the Domain

The chosen domain — Ghanaian government documents — has specific challenges:

1. **Acronym-heavy text:** NDC, NPP, NHIS, GES, SSNIT, GETFUND → query expansion solves this
2. **Structured tabular data (CSV):** Row-based chunking preserves relational integrity
3. **Long PDF with statistical tables:** Fixed-size chunking with overlap keeps numeric
   context together
4. **High hallucination risk:** Budget figures look plausible even when fabricated →
   strict "only use context" prompts and refusal instructions are essential
5. **Mixed data types:** Hybrid retrieval handles both keyword-specific electoral facts
   ("NDC in Volta region") and semantic policy concepts ("fiscal consolidation strategy")

---

## 5. Innovation: Query Expansion (Part G)

A custom expansion dictionary maps domain keywords to synonyms before the query
is embedded. This improves recall by 15–25% on acronym queries (measured in
Experiment G-1, G-2 in experiment_logs.md).

This is not a generic system — it is a **domain-specific** expansion map tuned
to Ghanaian political and fiscal vocabulary, making it a meaningful contribution
beyond a standard RAG implementation.

---

*David Owusu Appiah | Index: 10022300159 | CS4241 — AI — 2026*
