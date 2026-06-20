# Architecture

## Goal

Build a small, production-shaped RAG system for telecom specifications. V1 proves the hard part first: source-grounded answers over clause-structured standards documents.

## V1 Data Flow

```text
specs/manifest.example.yaml
  -> scripts/fetch_specs.py
  -> local ignored data directory
  -> ingestion parser
  -> clause-aware chunks
  -> citation metadata
  -> RAG Engine corpus
  -> Vector Search 2.0
  -> Cloud Run /ask endpoint
  -> cited answer
```

## V1 Components

### Spec Manifest

The manifest lists official public source URLs and expected document metadata. It is committed. Downloaded specifications are not committed.

### Fetcher

The fetcher downloads public source documents into an ignored local directory. It should verify checksums when available and record the fetched source URL.

### Ingestion

The ingestion pipeline parses each document, extracts section or clause labels, creates chunks, and attaches citation metadata.

Chunk metadata must include:

- `spec_id`
- `release`
- `version`
- `section`
- `page`
- `source_url`
- `chunk_hash`
- `doc_title`

### Retrieval and Generation

The serving path retrieves relevant chunks and generates answers that cite the supporting source. If the source is not retrieved, the system should say it cannot answer from the indexed corpus.

### Evaluation

V1 evaluation checks whether the expected supporting clause is retrieved, whether out-of-scope questions abstain, whether generated citations point to expected supporting clauses, and whether labeled answer assertions appear in grounded answers.

## Later Architecture

Later phases add:

- BigQuery structured lookup for exact parameter values.
- ADK or LangGraph agent routing between spec search and structured lookup.
- MCP server exposing project tools.
- Deeper observability with request tracing, tokens/sec, latency, and cost/request.
- Security hardening for customer or private corpora.

These are intentionally not part of V1 implementation.

## Deployment Shape

V1 deployment target:

- Cloud Run for the serving API.
- Google Cloud RAG Engine backed by Vector Search 2.0.
- Cloud Logging for structured request logs.
- Terraform for repeatable infrastructure.

Provision in `us-central1` for Vector Search 2.0-backed RAG Engine corpora.

## Failure Behavior

The API should prefer refusal over unsupported answers:

- If retrieval returns low-confidence or irrelevant chunks, answer with an insufficient-evidence response.
- If citation metadata is missing, do not fabricate a citation.
- If the source version is unknown, mark it unknown rather than guessing.

## Data Handling

The repository does not redistribute standards documents. Public source documents are fetched during setup and stored locally or in project-owned cloud storage.
