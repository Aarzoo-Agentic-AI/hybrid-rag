# Answer evaluation

- **date:** 2026-11-22 07:06 UTC
- **commit:** 6f2f011 (uncommitted changes)
- **golden set:** v1, 94 questions, sha256 3fae8871f553 (fact 37, table_lookup 12, exception 10, cross_policy 7, definition 7, follow_up 5, access 8, unanswerable 8)
- **index:** tenant `dummymocktest`, 106 distinct sections
- **embeddings:** `BAAI/bge-m3` (dense + sparse)
- **reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **machine:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.3
- **answer mode:** `hybrid_rerank`
- **llm:** `openai/gpt-5`, then `anthropic/claude-sonnet-5-5`
- **confidence gate:** min_rerank_score -12.0

## Summary

| Check | Result |
|---|---|
| Answerable: answered, cites an expected section, contains every fact | 85/86 |
| Answerable: answered | 85/86 |
| Answerable: cites an expected section | 85/86 |
| Answerable: contains every fact | 85/86 |
| Unanswerable: refused | 7/8 |
| Restricted questions asked without access: nothing revealed | 11/12 |
| Failed over to another provider | 0 |

## By category

| Category | Correct |
|---|---|
| fact | 36/37 |
| table_lookup | 12/12 |
| exception | 10/10 |
| cross_policy | 7/7 |
| definition | 7/7 |
| follow_up | 5/5 |
| access | 8/8 |
| unanswerable | 7/8 |

## Not correct

- **`ben-sabbatical`** (status not_found, no expected citation, missing facts: 7 years | seven years)
  - When do I get a sabbatical?
  - answer: The HR policies I found don't answer this question. Please contact the HR Service Center.
- **`na-germany-vacation`** (status answered)
  - How many vacation days do employees in Germany get?
  - answer: The policy lists PTO as 15 days (0–2 years), 20 days (3–5 years), and 25 days (6+ years), with up to 5 days allowed to roll over; this applies from 2027-01-01 [1]. The sources do not provide a Germany-specific vacation entitlement or any country-specific variation.  - For full leave details, the Handbook directs to the Employee Benefits Guide [2]. - Contact the HR Service Center for Germany-specif…
- **`merit-top-of-range`** (asked as employee, who may not read the sources, contains restricted facts: lump sum)
  - What happens to the raise of someone already at the top of their salary range?
  - answer: The policy only says annual merit increases are based on performance and position in the salary range; it does not specify what occurs if an employee is already at the top of the range (e.g., whether the increase is limited, deferred, or paid as a lump sum). Effective 2027-01-01 [1].  For guidance on your specific situation, please contact the HR Service Center.
