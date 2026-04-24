# src/prompt_builder.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part C: Prompt Engineering & Generation

Implements:
  1. Prompt template with context injection
  2. Hallucination control instructions
  3. Context window management (filter + truncate)
  4. Multiple prompt iterations for experimentation
"""

from __future__ import annotations

# ── Context window budget ─────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 6_000   # keep well within Claude's context window

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """\
You are an AI assistant for Academic City University with access to two \
knowledge sources:
  1. Ghana Election Results dataset
  2. Ghana 2025 Budget Statement

Answer the user's question using ONLY the information in the <context> block \
below. Do NOT invent facts. If the answer cannot be found in the context, say:
"I could not find relevant information in the provided documents."

Be concise, accurate, and cite which dataset your answer comes from when \
possible."""

SYSTEM_PROMPT_V2 = """\
You are a knowledgeable assistant for Academic City University, Ghana. \
You have been given excerpts from the Ghana Election Results and the \
Ghana 2025 Budget Statement.

Rules:
  - Answer ONLY from the provided context.
  - If information is absent, clearly state that.
  - Provide specific numbers, names, or percentages where available.
  - Do NOT hallucinate or guess.
  - At the end of your answer, indicate the source (election_csv or budget_pdf)."""

SYSTEM_PROMPT_V3 = """\
You are a precise, fact-based assistant for Academic City University.

STRICT RULES:
  1. Only use facts found in the <context> block.
  2. Never fabricate statistics, names, or dates.
  3. If the question cannot be answered from context, respond with:
     "This information is not available in the provided documents."
  4. Keep answers clear and structured.
  5. Mention confidence level: HIGH (direct match), MEDIUM (inferred), LOW (partial)."""


def build_prompt(
    query: str,
    chunks: list[dict],
    version: int = 2,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, str, str]:
    """
    Build the system prompt and user message for the LLM.

    Args:
        query:             The user's question.
        chunks:            Retrieved chunk dicts (with 'text', 'source', 'final_score').
        version:           Prompt template version (1, 2, or 3).
        max_context_chars: Hard limit on context character count.

    Returns:
        (system_prompt, user_message, context_used)
    """
    # 1. Select system prompt
    prompts = {1: SYSTEM_PROMPT_V1, 2: SYSTEM_PROMPT_V2, 3: SYSTEM_PROMPT_V3}
    system_prompt = prompts.get(version, SYSTEM_PROMPT_V2)

    # 2. Build context block (with truncation)
    context_parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        text   = chunk.get("text", "").strip()
        source = chunk.get("source", "unknown")
        score  = chunk.get("final_score", chunk.get("similarity_score", 0.0))
        header = f"[Chunk {i} | Source: {source} | Score: {score:.4f}]"
        entry  = f"{header}\n{text}"

        if total_chars + len(entry) > max_context_chars:
            # Truncate this chunk to fit within budget
            remaining = max_context_chars - total_chars - len(header) - 4
            if remaining > 200:
                entry = f"{header}\n{text[:remaining]}…"
                context_parts.append(entry)
            break

        context_parts.append(entry)
        total_chars += len(entry)

    context_used = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    # 3. User message
    user_message = f"<context>\n{context_used}\n</context>\n\nQuestion: {query}"

    return system_prompt, user_message, context_used


def build_comparison_prompts(query: str, chunks: list[dict]) -> dict[int, tuple]:
    """
    Return all three prompt versions for Part C experimentation.
    """
    return {
        v: build_prompt(query, chunks, version=v)
        for v in (1, 2, 3)
    }
