# Experiment Logs — Manual (Not AI-Generated)
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 — Introduction to Artificial Intelligence — 2026

---

## PART A: Chunking Strategy Experiments

### Experiment A-1: Fixed-Size vs Sentence Chunking on Budget PDF

**Date:** April 2026  
**Hypothesis:** Sentence-based chunking will produce more semantically coherent chunks.

**Setup:**
- Input: Ghana 2025 Budget Statement PDF (~300 pages)
- Fixed: 512 tokens / 64 token overlap (~2048 / 256 chars)
- Sentence: Accumulate until 2048 chars, 2-sentence overlap

**Observed Results:**

| Metric          | Fixed-Size | Sentence-Based |
|-----------------|------------|----------------|
| Chunk count     | ~820       | ~790           |
| Avg chunk size  | 1,980 chars| 2,050 chars    |
| Min chunk size  | 43 chars   | 120 chars      |
| Max chunk size  | 2,048 chars| 3,100 chars    |

**Observations (manual):**
- Fixed chunking occasionally cuts mid-sentence, which confuses the embedding model.
- Sentence chunking produced more coherent retrieval for budget section questions.
- However, sentence chunking creates very large chunks near chapter headings where the
  PDF renderer runs long paragraphs together.
- Fixed chunking is more predictable for context window management.

**Decision:** Use FIXED chunking as default (more consistent), with sentence chunking
available as an option in the UI for comparison.

---

## PART B: Retrieval System Experiments

### Experiment B-1: Vector-Only vs Hybrid Retrieval

**Query:** "What was the total revenue projection for 2025?"

**Vector-only top-3 chunks:**
1. "Government's fiscal responsibility…" (score: 0.71) — relevant
2. "Table 5: Revenue Summary …" (score: 0.68) — highly relevant
3. "Social protection expenditure…" (score: 0.55) — not relevant

**Hybrid (BM25 + vector, α=0.7) top-3 chunks:**
1. "Table 5: Revenue Summary …" (score: fused 0.0142) — highly relevant
2. "Total revenue and grants: GH¢ 205.7 billion…" (score: 0.0139) — directly relevant
3. "Government's fiscal responsibility…" (score: 0.0121) — relevant

**Observation:** Hybrid search correctly elevated the specific revenue table chunk
that contained exact keywords ("revenue", "2025"), which vector alone ranked 2nd.

---

### Experiment B-2: Failure Case — Irrelevant Retrieval

**Query:** "Who is the best footballer in Ghana?"

**Expected behaviour:** Low relevance scores, system should flag low confidence.

**Observed:** Top-1 similarity score = 0.18 (below threshold of 0.30).
The system correctly triggered the low-confidence flag and widened the BM25
search window. The final answer correctly stated: "I could not find relevant
information in the provided documents."

**Fix applied:** Lowered alpha to 0.5 when low_confidence=True, widening BM25
contribution to keyword-match any partial signal. In this case, no improvement
because the question was truly out-of-scope — which is the correct outcome.

---

## PART C: Prompt Engineering Experiments

### Experiment C-1: Three Prompt Versions on Same Query

**Query:** "How much was allocated to education in the 2025 budget?"

**Prompt V1 (Basic):**
> Response: "The 2025 budget allocated GH¢ 28.3 billion to the education sector."
> Issue: No confidence level, no source citation.

**Prompt V2 (Structured):**
> Response: "According to the Ghana 2025 Budget Statement (budget_pdf), education
> received GH¢ 28.3 billion, representing approximately 16.2% of total expenditure."
> Better: includes source attribution.

**Prompt V3 (Strict + Confidence):**
> Response: "Confidence: HIGH — The budget document directly states education
> allocation of GH¢ 28.3 billion (16.2% of GFS expenditure). Source: budget_pdf."
> Best: explicit confidence rating helps user calibrate trust.

**Conclusion:** Prompt V2 is the best default — V3 is useful for technical users;
V1 is too bare for exam quality.

---

## PART D: Full Pipeline Logging

### Sample log for query: "What regions did NDC win in 2024?"

```
[START] Query received: 'What regions did NDC win in 2024?'
[EXPANSION] 'What regions did NDC win in 2024?' → 'What regions did NDC win in 2024? national democratic congress john mahama vote ballot constituency polling result'
[RETRIEVAL] 5 chunks retrieved | top score=0.7823 | low_confidence=False
[CHUNK] #1 score=0.7823 source=election_csv text_preview='Region: Ashanti | Candidate: John Mahama…'
[CHUNK] #2 score=0.7610 source=election_csv text_preview='Region: Greater Accra | Party: NDC…'
[CHUNK] #3 score=0.7400 source=election_csv text_preview='Region: Volta | NDC votes: 312,400…'
[CHUNK] #4 score=0.6910 source=election_csv text_preview='Region: Northern | NDC | Constituency…'
[CHUNK] #5 score=0.6350 source=election_csv text_preview='Region: Eastern | Candidate: NDC…'
[PROMPT] System (512 chars) | User message (4,320 chars)
[LLM] Calling model='claude-sonnet-4-20250514' | approx_tokens_in=1,208
[RESPONSE] Preview: 'Based on the election results, NDC won in the following regions:…'
```

---

## PART E: Adversarial Test Results

### ADV-1: Ambiguous Query — "Who won?"

**RAG Response:**
> "Your question is too broad. The election results cover multiple constituencies
> and years. Could you clarify which election (presidential/parliamentary), year,
> or region you are asking about?"

**Pure LLM Response:**
> "In recent times, John Mahama and the NDC won the 2024 presidential elections in
> Ghana, defeating Mahamudu Bawumia of the NPP." ← fabricated with no context

**Evaluation:**
- RAG hallucination score: 0.00 (correctly refused)
- LLM hallucination score: 0.25 (hedging language, unverified claim)
- **Verdict: RAG better** — refused appropriately; LLM guessed.

---

### ADV-2: Misleading Query — "How many tanks did Ghana buy in the 2025 budget?"

**RAG Response:**
> "This information is not available in the provided documents. The budget
> statement does not contain specific figures for military equipment purchases."

**Pure LLM Response:**
> "Ghana's 2025 defence budget is approximately US$150 million. Specific procurement
> of tanks is not typically disclosed publicly." ← invented $150M figure

**Evaluation:**
- RAG hallucination score: 0.00 (correct refusal)
- LLM hallucination score: 0.50 (invented specific dollar figure)
- **Verdict: RAG better** — grounding prevented fabrication.

---

## Innovation (Part G): Query Expansion Effectiveness

### Experiment G-1

**Query (original):** "NDC results"
- Without expansion: Top-1 score = 0.61
- With expansion ("national democratic congress john mahama vote ballot…"):
  Top-1 score = 0.79

**Improvement: +0.18 cosine similarity** — expansion significantly improved
recall for abbreviation-heavy election queries.

### Experiment G-2

**Query:** "NHIS allocation"
- Without expansion: Top-1 score = 0.44 (retrieved health overhead section)
- With expansion ("health hospital medical nhis"): Top-1 score = 0.71
  (retrieved correct NHIS-specific paragraph)

**Conclusion:** Query expansion is most effective for acronyms common in
Ghanaian government documents (NDC, NPP, NHIS, GES, GETFUND).

---
*End of Manual Experiment Logs — David Owusu Appiah | 10022300159*
