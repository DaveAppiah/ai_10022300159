# src/rag_pipeline.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part D: Full RAG Pipeline

Flow:
  User Query → Query Expansion → Embedding → Hybrid Retrieval →
  Context Selection → Prompt Construction → Groq LLM → Response

Logging is injected at every stage via PipelineLogger.
"""

from __future__ import annotations
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from data_loader    import load_all_documents
from chunker        import chunk_documents
from embedder       import embed_query
from vector_store   import VectorStore
from retriever      import Retriever, expand_query
from prompt_builder import build_prompt
from logger         import PipelineLogger

# ── LLM Configuration ─────────────────────────────────────────────────────────
# Available Groq models (as of 2026):
# - llama-3.1-8b-instant     (recommended: fast, current, working)
# - gemma2-9b-it             (alternative: good performance)
# Note: mixtral-8x7b-32768 and llama-3.1-70b-versatile are decommissioned
LLM_MODEL  = "llama-3.1-8b-instant"   # current, working Groq model
MAX_TOKENS = 1024

# ── Index cache path ──────────────────────────────────────────────────────────
INDEX_PATH = "data/rag_index"


class RAGPipeline:
    """
    End-to-end RAG pipeline for the Academic City chat assistant.

    Call `build()` once to load data and build the index.
    Then call `query()` for each user question.
    """

    def __init__(self):
        self.vector_store: VectorStore | None = None
        self.retriever:    Retriever   | None = None
        self.chunks:       list[dict]         = []
        self._client:      Groq        | None = None
        self.is_ready:     bool               = False

    # ── Initialisation ─────────────────────────────────────────────────────────
    def build(self, chunking_strategy: str = "fixed") -> None:
        """
        Load data, chunk, embed, and build the FAISS index.
        Called once at startup (or after index cache miss).
        """
        import pathlib
        os.makedirs("data", exist_ok=True)

        if pathlib.Path(INDEX_PATH + ".faiss").exists():
            print("[pipeline] Loading cached index …")
            self.vector_store = VectorStore()
            self.vector_store.load(INDEX_PATH)
            self.chunks = self.vector_store.chunks
        else:
            print("[pipeline] Building index from scratch …")
            raw_docs = load_all_documents()
            print(f"[pipeline] Loaded {len(raw_docs)} raw documents "
                  f"(sources: {set(d['source'] for d in raw_docs)})")
            self.chunks = chunk_documents(raw_docs, strategy=chunking_strategy)
            print(f"[pipeline] Chunked into {len(self.chunks)} pieces "
                  f"(sources: {set(c['source'] for c in self.chunks)})")

            self.vector_store = VectorStore()
            self.vector_store.add_chunks(self.chunks)
            self.vector_store.save(INDEX_PATH)
            print("[pipeline] Index saved to disk.")

        self.retriever = Retriever(self.vector_store, self.chunks)
        self.is_ready  = True
        print(f"[pipeline] Ready. {len(self.chunks)} chunks indexed.")

    def _get_client(self) -> Groq:
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY", "")
            print(f"[DEBUG] API key found: {bool(api_key)} (length: {len(api_key)})")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            self._client = Groq(api_key=api_key)
            print("[DEBUG] Groq client created successfully")
        return self._client

    # ── Query ──────────────────────────────────────────────────────────────────
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        prompt_version: int = 2,
        use_expansion: bool = True,
        pure_llm_mode: bool = False,
    ) -> dict:
        """
        Run the full RAG pipeline for a single user query.

        Args:
            user_query:      The user's question.
            top_k:           Number of chunks to retrieve.
            prompt_version:  Which prompt template (1, 2, or 3).
            use_expansion:   Apply query expansion (Part G innovation).
            pure_llm_mode:   If True, skip retrieval and call LLM directly.

        Returns:
            dict with all pipeline outputs and logs.
        """
        logger = PipelineLogger(user_query)

        if not self.is_ready:
            return {"error": "Pipeline not initialised. Call build() first.", "log": logger.summary()}

        # ── STAGE 1: Query Expansion ───────────────────────────────────────────
        if use_expansion and not pure_llm_mode:
            expanded = expand_query(user_query)
            logger.log_expansion(user_query, expanded)
        else:
            expanded = user_query

        # ── STAGE 2: Retrieval ─────────────────────────────────────────────────
        retrieved_chunks: list[dict] = []
        retrieval_log: list[str]     = []
        low_conf = False

        if not pure_llm_mode:
            ret_result = self.retriever.retrieve(
                query=user_query,
                k=top_k,
                use_expansion=use_expansion,
            )
            retrieved_chunks = ret_result["results"]
            retrieval_log    = ret_result["retrieval_log"]
            low_conf         = ret_result["low_confidence"]

            top_score = retrieved_chunks[0].get("final_score", 0.0) if retrieved_chunks else 0.0
            logger.log_retrieval(len(retrieved_chunks), top_score, low_conf)
            logger.log_chunks(retrieved_chunks)

        # ── STAGE 3: Prompt Construction ──────────────────────────────────────
        if pure_llm_mode:
            system_prompt = "You are a helpful assistant."
            user_message  = user_query
            context_used  = ""
        else:
            system_prompt, user_message, context_used = build_prompt(
                user_query, retrieved_chunks, version=prompt_version
            )

        logger.log_prompt(system_prompt, user_message)

        # ── STAGE 4: LLM Call (Groq) ──────────────────────────────────────────
        approx_tokens = (len(system_prompt) + len(user_message)) // 4
        logger.log_llm_call(LLM_MODEL, approx_tokens)

        try:
            client = self._get_client()
            print(f"[DEBUG] Making LLM call with model: {LLM_MODEL}")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            answer = response.choices[0].message.content
            print(f"[DEBUG] LLM call successful, response length: {len(answer)}")
        except Exception as exc:
            print(f"[DEBUG] LLM call failed: {type(exc).__name__}: {exc}")
            logger.log_error(str(exc))
            answer = f"Error calling LLM: {exc}"

        logger.log_response(answer)

        return {
            "query":            user_query,
            "expanded_query":   expanded,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_log":    retrieval_log,
            "low_confidence":   low_conf,
            "system_prompt":    system_prompt,
            "user_message":     user_message,
            "context_used":     context_used,
            "answer":           answer,
            "pipeline_log":     logger.summary(),
            "prompt_version":   prompt_version,
            "pure_llm_mode":    pure_llm_mode,
        }


# ── Global singleton ──────────────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.build()
    return _pipeline
