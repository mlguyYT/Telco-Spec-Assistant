# Evaluation

V1 evaluation uses pure spec-retrieval questions over 3GPP TS 38.321, TS 38.322, and TS 38.331.

Metrics:

- retrieval recall@5
- per-spec recall@5 for MAC, RLC, and RRC answerable questions
- paraphrase recall@5 for deliberately hard wording variants
- non-paraphrase recall@5 for the original baseline rows
- answerable recall@5 for questions with expected MAC, RLC, or RRC clauses
- abstention accuracy for out-of-scope questions
- answer citation accuracy, measured by whether generated citations include an expected supporting clause
- answer refusal accuracy for out-of-scope questions
- answer quality over labeled questions, checked by required assertion terms in grounded extractive answers
- assertion group accuracy for precise value and terminology-variant questions
- p50 / p95 latency
- estimated cost per request

The multi-spec dataset has 176 rows. It combines the original RLC retrieval, precise-value, terminology-variant, out-of-scope, and paraphrase rows with broader MAC/RRC clause-retrieval coverage. The answerable rows cover 60 MAC questions, 39 RLC questions, and 70 RRC questions; the remaining rows are out-of-scope controls. The MAC and RRC counts include smoke retrieval rows for high-signal clauses. Rows with `required_answer_terms` are answer-level checks: every group must be present in the cited answer for the question to pass answer quality. These labels currently cover a representative subset, not every retrieval row.

Run locally after ingestion:

```bash
python -m eval.run --dataset eval/datasets/telco_retrieval_v1.jsonl --chunks .data/chunks/telco_v1.jsonl
```

Or run the full local Phase 1 pipeline:

```bash
python scripts/run_phase1_local.py --seed-dir ../input/3gpp-documents --no-download
```
