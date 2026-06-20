# MVP Scope

## Product Boundary

V1 is a narrow spec-RAG product over 3GPP TS 38.322, Radio Link Control.

The purpose is to prove that the system can ingest a real telecom standard, preserve citation metadata, retrieve relevant clauses, and produce grounded answers that a telecom engineer can verify.

## Included In V1

- Manifest-driven fetch from public source URLs.
- Local development support for a seed RLC document.
- Clause-aware chunking.
- Citation metadata schema.
- Google Cloud RAG Engine / Vector Search indexing path.
- Cloud Run API with `/ask` and `/health`.
- 41 retrieval, precise-value, terminology-variant, paraphrase, and out-of-scope evaluation questions across MAC, RLC, and RRC.
- Basic latency and cost measurement in eval output.
- Documentation of later structured lookup, agent, MCP, and observability phases.

## Excluded From V1

- BigQuery structured lookup implementation.
- ADK, LangGraph, or other agent implementation.
- MCP server implementation.
- Large multi-spec corpus.
- Production auth.
- Customer data.
- UI.
- VPC-SC, CMEK, or private networking implementation.

## Acceptance Criteria

- A local developer can fetch or provide the seed RLC spec without committing it.
- Ingestion produces chunks with spec, release, version, section, source URL, and hash metadata.
- `/ask` returns answers with citations for questions covered by the indexed RLC corpus.
- `/ask` refuses or states insufficient evidence when the corpus does not support an answer.
- Eval runner reports recall@5, paraphrase recall@5, abstention accuracy, citation support, and answer assertion quality for the RLC question set.
- No downloaded specifications or secrets are tracked by git.

## Human Decisions Before Implementation

- Confirm the official public source URL for the `38322-j20` RLC document.
- Choose the first generation model variant.
- Decide whether the first implementation should support local-only retrieval before cloud indexing.
