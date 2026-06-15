# Evaluation

V1 evaluation uses pure spec-retrieval questions over 3GPP TS 38.322.

Metrics:

- retrieval recall@5
- citation support
- groundedness
- p50 / p95 latency
- estimated cost per request

Run locally after ingestion:

```bash
python -m eval.run --dataset eval/datasets/rlc_retrieval_v1.jsonl --chunks .data/chunks/rlc_v1.jsonl
```
