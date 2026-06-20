# Serving

V1 serving target:

- `GET /health`
- `POST /ask`

The `/ask` endpoint returns grounded answers with citations or states that the indexed corpus does not contain enough evidence. The default generator is conservative and extractive. A Gemini-backed generator can be enabled explicitly for cloud runs.

Run the local baseline after chunks have been generated:

```bash
python -m serving.app --chunks .data/chunks/telco_v1.jsonl
curl -X POST http://127.0.0.1:8080/ask \
  -H 'content-type: application/json' \
  -d '{"q":"What are the three RLC modes?"}'
```

The local baseline extracts short evidence lines from the strongest retrieved clauses, which keeps unsupported claims out of the API and keeps CI credential-free.

To use Gemini generation, install the optional cloud dependencies, configure Vertex AI credentials, and set:

```bash
GENERATOR=gemini
GEMINI_MODEL=<vertex-ai-gemini-model-id>
```

The Gemini path receives only retrieved chunks and must return JSON with supported status, answer text, and citation IDs from those chunks. If the model does not provide valid retrieved citation IDs, the API refuses the answer.

Runtime settings can be provided through flags or environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CHUNKS_PATH` | `.data/chunks/telco_v1.jsonl` | Local chunk JSONL file |
| `GENERATOR` | `extractive` | Answer backend: `extractive` or `gemini` |
| `GEMINI_MODEL` | unset | Required only when `GENERATOR=gemini` |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8080` | HTTP port |
| `TOP_K` | `5` | Retrieval depth |
| `MIN_SCORE` | `auto` | Minimum top retrieval score. Auto keeps BM25 at `1.0` and uses `0.0` for rank/vector retrievers. |

Run the same API in a local container without baking generated chunks into the image:

```bash
docker build -t telco-spec-assistant .
docker run --rm -p 8080:8080 \
  -v "$PWD/.data/chunks:/data/chunks:ro" \
  telco-spec-assistant
```
