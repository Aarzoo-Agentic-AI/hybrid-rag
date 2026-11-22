# Answer evaluation

- **date:** 2026-11-22 09:00 UTC
- **commit:** e82c9a5 (uncommitted changes)
- **golden set:** v1, sha256 7ce3a1d7985a
- **index:** tenant `dummymocktest`, 106 distinct sections
- **embeddings:** `BAAI/bge-m3` (dense + sparse)
- **reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **machine:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.3
- **answer mode:** `hybrid_rerank`
- **strategy:** `agentic`
- **sources sent:** top 8 chunks, at most 4000 tokens
- **llm:** `openai/gpt-5`, then `anthropic/claude-sonnet-5-5`
- **confidence gate:** min_rerank_score -12.0
- **rescored:** 2026-11-22 09:17 UTC (same answers)

## Summary

| Check | Result |
|---|---|
| Answerable: answered, cites an expected section, contains every fact | 86/86 |
| Answerable: answered | 86/86 |
| Answerable: cites an expected section | 86/86 |
| Answerable: contains every fact | 86/86 |
| Unanswerable: refused | 7/8 |
| Restricted questions asked without access: nothing revealed | 11/12 |
| Failed over to another provider | 0 |
| Searches per question (mean) | 2.03 |
| Latency per question (median) | 7.7 s |

## By category

| Category | Correct |
|---|---|
| fact | 37/37 |
| table_lookup | 12/12 |
| exception | 10/10 |
| cross_policy | 7/7 |
| definition | 7/7 |
| follow_up | 5/5 |
| access | 8/8 |
| unanswerable | 7/8 |

## Not correct

- **`na-germany-vacation`** (status answered)
  - How many vacation days do employees in Germany get?
  - answer: The provided policies don’t state a Germany-specific vacation allotment. Company PTO is 15 days (0–2 years), 20 days (3–5 years), and 25 days (6+ years), with up to 5 days allowed to roll over; this applies from 2027-01-01 [1].  What the policies don’t cover: Any Germany-specific statutory or country-adjusted vacation rules beyond the company PTO tiers. For country-specific details, please contact…
- **`merit-def-performance-year`** (asked as employee, who may not read the sources, contains restricted facts: December 31 | Dec 31 | 31 December)
  - What dates does the performance year cover?
  - answer: January 1 through December 31 (calendar year), with goals set in January and the annual review in December; calibration occurs after reviews [1]. This framework applies starting January 1, 2027 [5].
