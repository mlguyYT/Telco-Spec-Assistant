# Evaluation

V1 evaluation uses pure spec-retrieval questions over 3GPP TS 38.321, TS 38.322, and TS 38.331.

Metrics:

- retrieval recall@5
- paraphrase recall@5 for deliberately hard wording variants
- non-paraphrase recall@5 for the original baseline rows
- answerable recall@5 for questions with expected RLC clauses
- abstention accuracy for out-of-scope questions
- citation support, approximated by expected-section hits in the local baseline
- answer quality over labeled questions, checked by required assertion terms in grounded extractive answers
- assertion group accuracy for precise value and terminology-variant questions
- p50 / p95 latency
- estimated cost per request

The RLC questions measure retrieval coverage, precise field values, terminology variants, out-of-scope controls, and answerable paraphrase cases. The multi-spec dataset keeps those RLC rows and adds initial MAC/RRC clause-retrieval rows. Rows with `required_answer_terms` are answer-level checks: every group must be present in the cited answer for the question to pass answer quality.

Run locally after ingestion:

```bash
python -m eval.run --dataset eval/datasets/telco_retrieval_v1.jsonl --chunks .data/chunks/telco_v1.jsonl
```

Or run the full local Phase 1 pipeline:

```bash
python scripts/run_phase1_local.py --seed-dir ../input/3gpp-documents --no-download
```
