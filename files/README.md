# AI RAG Chat Assistant — Academic City University

**Student Name:** David Owusu Appiah  
**Index Number:** 10022300159  
**Course:** CS4241 — Introduction to Artificial Intelligence  
**Semester:** Second Semester, 2026  
**Lecturer:** Godwin N. Danso  

---

## Project Overview

A fully custom-built **Retrieval-Augmented Generation (RAG)** chat assistant for Academic City University, built **without** LangChain, LlamaIndex, or any pre-built RAG pipelines.

The assistant answers questions grounded in two datasets:
- Ghana Election Results (CSV)
- Ghana 2025 Budget Statement (PDF)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Expansion │  ← novel feature: multi-term expansion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding Engine │  ← sentence-transformers (manual)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │  ← custom FAISS index (manual build)
│     Store       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Hybrid Retrieval       │  ← vector + BM25 keyword search
│  (Top-K + Re-ranking)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Prompt Builder  │  ← context injection + hallucination control
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq LLM API    │  ← Groq llama-3.1-8b-instant
└────────┬────────┘
         │
         ▼
    Final Response
```

---

## Setup & Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Groq API Key:**
   - Get your free Groq API key from: https://console.groq.com/keys
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

3. **Run the application:**
```bash
streamlit run app.py
```

The app will **automatically build the vector index** on the first load. No manual build button required.

### Environment Variables
The project uses the following environment variable:
- `GROQ_API_KEY` - Your Groq API key (required for LLM functionality)

---

## CI/CD

This project uses GitHub Actions for continuous integration:

- **CI Workflow** (`/.github/workflows/ci.yml`): Runs on every push/PR
  - Linting (flake8, black, isort, mypy)
  - Import validation
  - Dependency installation checks

- **Deploy Prep Workflow** (`/.github/workflows/deploy.yml`): Runs on main branch
  - Verifies Streamlit app imports
  - Checks data directory presence
  - Validates environment template

---

## File Structure

```
ai_10022300159/
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py      # Part A: Data loading & cleaning
│   ├── chunker.py          # Part A: Chunking strategies
│   ├── embedder.py         # Part B: Embedding pipeline
│   ├── vector_store.py     # Part B: FAISS vector store
│   ├── retriever.py        # Part B: Top-K retrieval + hybrid search
│   ├── prompt_builder.py   # Part C: Prompt engineering
│   ├── rag_pipeline.py     # Part D: Full pipeline
│   ├── evaluator.py        # Part E: Evaluation & adversarial tests
│   └── logger.py           # Stage-by-stage logging
├── logs/
│   └── experiment_logs.md  # Manual experiment logs
└── docs/
    └── architecture.md     # Architecture documentation
```

---

## Deployment

- **GitHub:** https://github.com/DavidOwusuAppiah/ai_10022300159  
- **Live App:** (Streamlit Cloud / Render URL here after deployment)

---

## Key Design Decisions

1. **Chunking:** Fixed-size (512 tokens, 64 overlap) for budget PDF; row-based for CSV
2. **Embeddings:** `all-MiniLM-L6-v2` — fast, accurate, free
3. **Vector Store:** FAISS IndexFlatIP (inner-product / cosine similarity)
4. **Retrieval:** Hybrid BM25 + vector, top-5, with cross-encoder re-ranking
5. **Innovation:** Query expansion using synonym generation before retrieval
6. **Hallucination Control:** Strict "only use context" prompt + confidence thresholds

---

*David Owusu Appiah | 10022300159*
