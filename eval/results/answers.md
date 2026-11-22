# Answer evaluation

- **date:** 2026-11-22 08:43 UTC
- **commit:** e82c9a5 (uncommitted changes)
- **golden set:** v1, sha256 7ce3a1d7985a
- **index:** tenant `dummymocktest`, 106 distinct sections
- **embeddings:** `BAAI/bge-m3` (dense + sparse)
- **reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **machine:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.3
- **answer mode:** `hybrid_rerank`
- **strategy:** `single`
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
| Unanswerable: refused | 8/8 |
| Restricted questions asked without access: nothing revealed | 12/12 |
| Failed over to another provider | 0 |
| Searches per question (mean) | 1.00 |
| Latency per question (median) | 4.9 s |

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
| unanswerable | 8/8 |

## Not correct

None.
