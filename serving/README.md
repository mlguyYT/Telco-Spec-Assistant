# Serving

V1 serving target:

- `GET /health`
- `POST /ask`

The `/ask` endpoint should return grounded answers with citations or state that the indexed corpus does not contain enough evidence.

Run the local baseline after chunks have been generated:

```bash
python -m serving.app --chunks .data/chunks/rlc_v1.jsonl
curl -X POST http://127.0.0.1:8080/ask \
  -H 'content-type: application/json' \
  -d '{"q":"What are the three RLC modes?"}'
```
