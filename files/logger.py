# src/logger.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part D: Stage-by-stage pipeline logging.

Logs each RAG pipeline stage to:
  - In-memory list (returned to UI for display)
  - Optionally a file (logs/pipeline.log)
"""

from __future__ import annotations
import datetime
import pathlib

LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PipelineLogger:
    """Collects log entries for one query round-trip."""

    def __init__(self, query: str):
        self.query = query
        self.entries: list[dict] = []
        self._log("START", f"Query received: '{query}'")

    def _log(self, stage: str, message: str) -> None:
        entry = {"ts": _timestamp(), "stage": stage, "message": message}
        self.entries.append(entry)
        # Also write to file
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{entry['ts']}] [{stage}] {message}\n")

    # ── Stage loggers ──────────────────────────────────────────────────────────
    def log_expansion(self, original: str, expanded: str) -> None:
        self._log("EXPANSION", f"'{original}' → '{expanded}'")

    def log_retrieval(self, n_chunks: int, top_score: float, low_conf: bool) -> None:
        self._log(
            "RETRIEVAL",
            f"{n_chunks} chunks retrieved | top score={top_score:.4f} | "
            f"low_confidence={low_conf}"
        )

    def log_chunks(self, chunks: list[dict]) -> None:
        for i, c in enumerate(chunks, 1):
            score = c.get("final_score", c.get("similarity_score", 0.0))
            self._log(
                "CHUNK",
                f"#{i} score={score:.4f} source={c.get('source','?')} "
                f"text_preview='{c.get('text','')[:80].replace(chr(10),' ')}…'"
            )

    def log_prompt(self, system_prompt: str, user_message: str) -> None:
        self._log("PROMPT", f"System ({len(system_prompt)} chars) | "
                            f"User message ({len(user_message)} chars)")

    def log_llm_call(self, model: str, tokens_in: int = 0) -> None:
        self._log("LLM", f"Calling model='{model}' | approx_tokens_in={tokens_in}")

    def log_response(self, response_preview: str) -> None:
        self._log("RESPONSE", f"Preview: '{response_preview[:120].replace(chr(10),' ')}…'")

    def log_error(self, error: str) -> None:
        self._log("ERROR", error)

    def summary(self) -> list[str]:
        return [f"[{e['stage']}] {e['message']}" for e in self.entries]
