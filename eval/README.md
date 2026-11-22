# Evaluation: the golden dataset

`golden.yaml` holds 94 HR questions, each labelled with the policy sections that answer it. `evaluate.py` runs them through retrieval in every mode and, with `--answers`, through the full answer service. The results are in [`results/`](results/), and the main table is in the [project README](../README.md#evaluation-results).

## Running it

```bash
uv run python eval/evaluate.py                       # retrieval, all four modes: ~5 minutes, no LLM
uv run python eval/evaluate.py --answers             # also answer every question (~110 OpenAI calls)
uv run python eval/evaluate.py --only trv-hotel-boston --only na-ceo --mode hybrid_rerank
uv run python eval/evaluate.py --reranker BAAI/bge-reranker-v2-m3 --out eval/results/bge-v2-m3
uv run python eval/evaluate.py --answers --no-retrieval --min-rerank-score off --out /tmp/gate-off
uv run python eval/evaluate.py --answers --no-retrieval --strategy agentic --out eval/results/agentic
uv run python eval/evaluate.py --rescore eval/results    # after fixing a fact: re-check saved answers, no LLM
```

- **What it needs:** Qdrant with `data/pdfs` indexed (`scripts/ingest.py`). It reads the same `config.yaml` and `.env` as the API.
- **Who asks:** each question is asked as a demo principal (`employee`, `manager`, `hr-comp`, `admin`), the same as `scripts/make_api_key.py --demo`.
- **Checked first:** every expected section must exist in the index. A typo or a missing document stops the run, so it can't quietly score as a miss.
- **Tracing:** LangSmith tracing follows `.env`. Set `LANGSMITH_TRACING=false` to keep evaluation runs out of your project.
- **Output:**
  - `retrieval.md` and `retrieval.json`: every mode's metrics and per-question ranks.
  - `answers.md` and `answers.json`: every answer, its status, citations and facts.

## What is measured

**Retrieval** (no LLM), per mode:

| Metric | Meaning |
|---|---|
| Recall@k | The share of a question's expected sections found in the top k chunks, averaged over questions |
| MRR | 1 / the rank of the first chunk that meets any requirement, averaged |
| Median latency | Per search, with a fresh query embedding (the embedding cache is cleared each time) |

- **Per chunk:** ranks are per chunk, as retrieval returns them to the LLM (top 10, without small-to-big expansion).
- **Alternatives:** a requirement can be met by any of several sections, because the same rule is often stated in two policies (the $75 meal limit is in the Handbook, the Travel Policy and the Expense Policy).
- **Breakdowns:** by category and by question style.
- **Leakage:** every question with `hidden_from` is asked again, in every mode, by a caller who may not read its sources. Any chunk from those sources is a failure.
- **Confidence gate:** the best reranker score of answerable versus unanswerable questions, and how many of each a `min_rerank_score` threshold would keep or refuse.

**Answers** (`--answers`), in `retrieval.mode`:
- **Answerable questions:** answered, citing at least one expected section, and containing every `facts` entry.
- **Unanswerable questions:** refused (`not_found`, `low_confidence` or `no_sources`).
- **Restricted questions asked without access:** nothing cited from the restricted sources, and none of their facts in the answer.

These are string checks, not an LLM judge. They are cheap, repeatable and strict: a correct answer that words a number differently fails, so read `answers.md` before trusting a low score.

## The dataset

| Category | n | What it tests |
|---|---|---|
| `fact` | 37 | One rule from one section |
| `table_lookup` | 12 | A value in a table row (lodging tiers, approval limits, PTO by tenure, plan costs) |
| `exception` | 10 | Special cases: people managers' notice period, part-time eligibility, long stays, protected leave |
| `cross_policy` | 7 | Rules stated in two policies; the Handbook governs where they differ. Several sections are expected |
| `definition` | 7 | Glossary and definition entries |
| `follow_up` | 5 | A short follow-up after earlier turns. Retrieval is scored on a hand-written standalone form; `--answers` uses the real rewrite |
| `access` | 8 | The restricted merit and bonus policy, asked by a manager or HR compensation, and again by an employee |
| `unanswerable` | 8 | Things the policies don't cover. Some are near misses: pet insurance exists but says nothing about dogs at work |

**Question style:** 41 questions use the policies' own words (`keyword`) and 45 are deliberately reworded (`paraphrase`, for example "Can my partner come along on a work trip at company cost?" for "Spouse, partner or family travel costs" are non-reimbursable). The questions were written by reading the policies, which tends to copy their wording and favour keyword search; the paraphrased half measures that.

**Not covered:** questions about country differences and superseded versions. All five policies are current and US-only, so there is nothing to retrieve; the `jurisdiction` and `current_only` filters behind them are tested in `tests/test_retrieval.py`.

## Adding a question

```yaml
- id: trv-hotel-boston                 # <policy>-<topic>, lowercase
  question: "What's the hotel limit for a trip to Boston?"
  category: table_lookup
  style: keyword
  caller: employee                     # optional, the default
  expected: ["trv::6 Lodging"]         # alias::section path, exactly as indexed
  facts: ["$300"]                      # optional; "a|b" for alternatives
```

- **Section paths:** `uv run python scripts/search.py "your question" --as admin` shows them (`§6 Lodging` in the citation is the path `6 Lodging`; nested ones look like `2 Medical Plans > Medical plan features`).
- **Alternatives and several sections:** `expected: [["hb::20 Travel and Expenses", "trv::8 Meals"]]` is one requirement either section meets; `expected: ["hb::20 Travel and Expenses", "trv::8 Meals"]` needs both.
- **Facts:** keep them to an exact figure or phrase from the policy ("$300", "16 weeks"). Avoid short words like "no", which match inside other words.
- **Checks:** `uv run pytest tests/test_evaluation.py` validates the file, including that every question about the restricted policy is also checked against a caller without access.
