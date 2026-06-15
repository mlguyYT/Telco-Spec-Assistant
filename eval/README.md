# Evaluation

V1 evaluation uses pure spec-retrieval questions over 3GPP TS 38.322.

Metrics:

- retrieval recall@5
- answerable recall@5 for questions with expected RLC clauses
- abstention accuracy for out-of-scope questions
- citation support, approximated by expected-section hits in the local baseline
- answer quality over labeled questions, checked by required assertion terms in grounded extractive answers
- assertion group accuracy for precise value and terminology-variant questions
- p50 / p95 latency
- estimated cost per request

The first 20 questions measure retrieval coverage over RLC clauses. The final 6 harden the suite with precise field-value questions, terminology variants, and out-of-scope controls. Rows with `required_answer_terms` are answer-level checks: every group must be present in the cited answer for the question to pass answer quality.

Run locally after ingestion:

```bash
python -m eval.run --dataset eval/datasets/rlc_retrieval_v1.jsonl --chunks .data/chunks/rlc_v1.jsonl
```

Or run the full local Phase 1 pipeline:

```bash
python scripts/run_phase1_local.py --seed-dir ../input/3gpp-documents --no-download
```
