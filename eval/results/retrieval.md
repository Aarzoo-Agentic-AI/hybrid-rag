# Retrieval evaluation

- **date:** 2026-11-22 07:06 UTC
- **commit:** 6f2f011 (uncommitted changes)
- **golden set:** v1, 94 questions, sha256 3fae8871f553 (fact 37, table_lookup 12, exception 10, cross_policy 7, definition 7, follow_up 5, access 8, unanswerable 8)
- **index:** tenant `dummymocktest`, 106 distinct sections
- **embeddings:** `BAAI/bge-m3` (dense + sparse)
- **reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **machine:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.3
- **retrieval:** top 10 chunks; hybrid prefetch 50 dense + 50 sparse; reranker on 20 fused candidates; no section expansion

## Overall

86 answerable questions, each asked as its `caller`. A question with several expected sections scores the share it found. Ranks are per chunk, as sent to the LLM.

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median latency |
|---|---|---|---|---|---|---|
| `dense` | 77.3% | 94.8% | 98.8% | 98.8% | 0.878 | 0.20 s |
| `sparse` | 50.6% | 70.3% | 77.9% | 86.0% | 0.634 | 0.21 s |
| `hybrid` | 62.8% | 90.7% | 95.9% | 100% | 0.787 | 0.21 s |
| `hybrid_rerank` | 82.6% | 95.3% | 98.3% | 100% | 0.908 | 0.92 s |

## Recall@5 by category

| Category | n | `dense` | `sparse` | `hybrid` | `hybrid_rerank` |
|---|---|---|---|---|---|
| fact | 37 | 97.3% | 70.3% | 91.9% | 97.3% |
| table_lookup | 12 | 100% | 91.7% | 100% | 100% |
| exception | 10 | 100% | 60.0% | 100% | 100% |
| cross_policy | 7 | 100% | 71.4% | 92.9% | 92.9% |
| definition | 7 | 100% | 100% | 100% | 100% |
| follow_up | 5 | 100% | 80.0% | 100% | 100% |
| access | 8 | 100% | 100% | 100% | 100% |

## Recall@5 by question style

| Question style | n | `dense` | `sparse` | `hybrid` | `hybrid_rerank` |
|---|---|---|---|---|---|
| keyword | 41 | 100% | 91.5% | 100% | 97.6% |
| paraphrase | 45 | 97.8% | 65.6% | 92.2% | 98.9% |

## Access: restricted sources

12 restricted questions were also asked by a caller who may not read their sources, in every mode (48 searches). **No chunk from a restricted source was returned.**

## Confidence gate (`hybrid_rerank`)

The answer service skips the LLM when the best reranker score is below `answer.min_rerank_score`. *Kept*: answerable questions that still reach the LLM. *Refused*: unanswerable questions the gate stops (the LLM can still refuse the rest with its not-found answer).

| Threshold | Kept | Refused |
|---|---|---|
| -12 (current) | 100% | 0% |
| -11 | 96.5% | 12.5% |
| -10 | 94.2% | 25.0% |
| -9 | 91.9% | 37.5% |
| -8 | 87.2% | 37.5% |
| -7 | 82.6% | 50.0% |
| -6 | 77.9% | 75.0% |
| -5 | 72.1% | 75.0% |
| -4 | 66.3% | 87.5% |
| -3 | 64.0% | 100% |
| -2 | 59.3% | 100% |
| -1 | 55.8% | 100% |
| 0 | 52.3% | 100% |

Best balanced threshold: **-3.72** (kept 65.1%, refused 100%).
Lowest answerable score: -11.13; highest unanswerable score: -3.76.

Unanswerable questions, best score:

- `na-ceo`: -9.23 (Document Control)
- `na-dogs-in-office`: -11.22 (14 Health, Safety and Security)
- `na-free-lunch`: -10.54 (8 Wellbeing and Support Programs)
- `na-germany-vacation`: -4.07 (7 Time Off and Leave)
- `na-relocation`: -3.76 (6 Retirement Savings: 401(k))
- `na-stock-option-vesting`: -6.91 (6 Retirement Savings: 401(k))
- `na-tuition-payback`: -6.24 (9 Additional and Voluntary Benefits > Company-paid)
- `na-vpn-reset`: -7.69 (12 Health, safety and security)

## Misses in `hybrid_rerank` (Recall@5 below 100%)

- **`ben-sabbatical`** (fact): When do I get a sabbatical?
  - expected: Employee_Benefits_Guide_2027.pdf::7 Time Off and Leave (found at #7)
  - top 3: HR_Policy_Handbook_2027.pdf::23 Separation from Employment, Employee_Benefits_Guide_2027.pdf::1 Eligibility and Enrollment > When you can enroll, HR_Policy_Handbook_2027.pdf::13 Leave and Time Off
- **`xp-home-office`** (cross_policy): What can I get for setting up a home office, and do I claim it as an expense?
  - expected: Employee_Benefits_Guide_2027.pdf::9 Additional and Voluntary Benefits > Company-paid; Expense_Policy.pdf::4 Eligible expense categories (found at #4, #7)
  - top 3: Expense_Policy.pdf::2 Principles, Expense_Policy.pdf::3 Roles and responsibilities, Expense_Policy.pdf::10 Fraud, audit and consequences
