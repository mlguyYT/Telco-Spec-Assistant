# Scripts

Utility scripts for fetching public specifications, preparing local development data, and optionally testing managed vector retrieval.

Local Phase 1:

- `fetch_specs.py --manifest specs/manifest.example.yaml`
- `fetch_specs.py --manifest specs/manifest.example.yaml --seed-dir ../input/3gpp-documents --no-download`
- `run_phase1_local.py --seed-dir ../input/3gpp-documents --no-download`

Optional Vertex AI Vector Search path:

1. Install optional cloud dependencies:

   ```bash
   pip install -r requirements-cloud.txt
   ```

2. Authenticate and configure `.env` locally. Keep `RETRIEVER=bm25` until the endpoint is created:

   ```bash
   gcloud auth application-default login
   ```

3. Embed the already-generated clause chunks:

   ```bash
   python scripts/embed_chunks.py
   ```

4. Create and deploy the Vector Search index only when ready to run a paid test window:

   ```bash
   python scripts/create_vector_index.py
   ```

   The command prints `VS_INDEX_ID`, `VS_ENDPOINT_ID`, and `VS_DEPLOYED_INDEX_ID` values for local `.env`.

5. Compare retrievers:

   ```bash
   RETRIEVER=vertex python scripts/compare_retrievers.py
   ```

   The default comparison runs `bm25`, `vertex`, and `hybrid` against `.data/chunks/telco_v1.jsonl` and `eval/datasets/telco_retrieval_v1.jsonl`. The hybrid retriever uses Reciprocal Rank Fusion over BM25 and Vertex results, merging by chunk ID instead of comparing raw scores. `HYBRID_VERTEX_WEIGHT` applies a small rank-only weight to the vector retriever for paraphrase-heavy queries.

6. Tear down the deployed endpoint when finished:

   ```bash
   python scripts/delete_vector_index.py
   ```

Cost note: embedding requests are usage-based. A deployed Vector Search index endpoint can bill while it is running, so create and delete it in the same work session.

Downloaded specifications must be written to an ignored local data directory and never committed.
