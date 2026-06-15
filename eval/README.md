# Evaluation

V1 evaluation uses pure spec-retrieval questions over 3GPP TS 38.322.

Metrics:

- retrieval recall@5
- answerable recall@5 for questions with expected RLC clauses
- abstention accuracy for out-of-scope questions
- citation support, approximated by expected-section hits in the local baseline
- answer-level groundedness, added once generation is enabled and gold answers are checked
- p50 / p95 latency
- estimated cost per request

Run locally after ingestion:

```bash
python -m eval.run --dataset eval/datasets/rlc_retrieval_v1.jsonl --chunks .data/chunks/rlc_v1.jsonl
```
