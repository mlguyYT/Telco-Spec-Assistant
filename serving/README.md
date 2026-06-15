# Serving

V1 serving target:

- `GET /health`
- `POST /ask`

The `/ask` endpoint should return conservative extractive answers with citations or state that the indexed corpus does not contain enough evidence.

Run the local baseline after chunks have been generated:

```bash
python -m serving.app --chunks .data/chunks/rlc_v1.jsonl
curl -X POST http://127.0.0.1:8080/ask \
  -H 'content-type: application/json' \
  -d '{"q":"What are the three RLC modes?"}'
```

The local baseline does not use a generative model yet. It extracts short evidence lines from the strongest retrieved clauses, which keeps unsupported claims out of the API while retrieval quality is still being tuned.
